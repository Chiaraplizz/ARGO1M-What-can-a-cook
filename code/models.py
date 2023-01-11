import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
import numpy as np
from torch.nn.init import normal_
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder
import einops
from utils import get_seq_len_from_config, get_feat_dim_from_config
from torch.autograd import Function
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
# import clip
import time
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict
from sentence_transformers import SentenceTransformer, util


class TextModel(nn.Module):
    def __init__(self, config, dataset=None):
        super().__init__()
        self.model = SentenceTransformer('sentence-transformers/' + config.text_model)
        self.text_dim = config.text_dim
        self.dim = config.clip_dim
        self.config = config
        if self.config.one_proj:
            # self.text_proj = nn.Linear(self.dim, self.dim)
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_dim, self.dim),
                nn.ReLU(True),
                nn.Linear(self.dim, self.dim)
            )


    def forward(self, text):
        if self.config.one_proj:
            return {'text_feat': self.text_proj(self.model.encode(text, convert_to_tensor=True))}
        else:
            return {'text_feat': self.model.encode(text, convert_to_tensor=True)}


class GenerativeCaption(nn.Module):
    """
    Generative caption loss.
    """

    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda')
        self.lm = T5ForConditionalGeneration.from_pretrained(self.config.gen_model).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.gen_model)
        self.softmax = torch.nn.Softmax(dim=0)

        # self.layer_norm = torch.nn.LayerNorm(512)

    def forward(self, output, narration, target, is_test=False):

        # needs access to text, scenario labels and representions
        # v_reps = self.layer_norm(output["representations"])
        v_reps = F.normalize(output).to(self.device)
        labels = target[:, 0]
        scenarios = target[:, 1]

        target_encoding = self.tokenizer(narration, padding="longest", max_length=30, truncation=True,
                                         return_tensors='pt').input_ids.to(self.device)

        if self.config.gen_attn == "self":
            attn_v_reps = v_reps

        else:
            sim = torch.matmul(v_reps, v_reps.t())

            if self.config.gen_attn == "all":
                sim_sm = self.softmax(sim)
            elif self.config.gen_attn == "other":
                # Set elements to 0 with no similarity. Two different methods.
                # mask = 1 - torch.eye(sim.shape[0])
                # sim_sm = self.softmax(sim) * mask
                sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
                sim_sm = self.softmax(sim)
            elif self.config.gen_attn == "other scenarios":
                sim = sim - 1000 * torch.stack(
                    [torch.where(scenarios == scenarios[i], 1, 0) for i in range(scenarios.shape[0])])
                sim_sm = self.softmax(sim)

            elif self.config.gen_attn == "few":
                sim = sim - 1000 * torch.stack(
                    [(torch.rand(size=(1, self.config.batch_size), device=sim.device) < 0.75).int() for i in
                     range(scenarios.shape[0])]).reshape(self.config.batch_size, self.config.batch_size)
                sim_sm = self.softmax(sim)
            elif self.config.gen_attn == "one scenario":
                scenario_list = [random.randint(0, self.config.n_domains) for i in range(self.config.batch_size)]
                sim = sim - 1000 * torch.stack(
                    [torch.where(scenarios != scenario_list[i], 1, 0) for i in range(scenarios.shape[0])])
                sim_sm = self.softmax(sim)
            elif self.config.gen_attn == "same label":
                sim = sim - 1000 * torch.stack(
                    [torch.where(scenarios == scenarios[i], 1, 0) for i in range(scenarios.shape[0])])
                sim = sim - 1000 * torch.stack(
                    [torch.where(labels == labels[i], 0, 1) for i in range(labels.shape[0])])
                sim_sm = self.softmax(sim)
            elif self.config.gen_attn == "same scenario":
                sim = sim - 1000 * torch.stack(
                    [torch.where(scenarios == scenarios[i], 0, 1) for i in range(scenarios.shape[0])])
                sim_sm = self.softmax(sim)
            elif self.config.gen_attn == "one scenario zero":
                scenario_list = [random.randint(0, self.config.n_domains) for i in range(self.config.batch_size)]
                sim = sim - 1000 * torch.stack(
                    [torch.where(scenarios != scenario_list[i], 1, 0) for i in range(scenarios.shape[0])])
                sim_sm = self.softmax(sim) * torch.stack(
                    [torch.where(scenarios != scenario_list[i], 0, 1) for i in range(scenarios.shape[0])])

            attn_v_reps = torch.matmul(sim_sm, v_reps)

        # TODO: change the 1 to be the number of transformer inputs
        # if is_test:
        #    attn_v_reps = v_reps
        attn_v_reps = [einops.rearrange(attn_v_reps, 'b d -> b 1 d')]
        gen_caption = self.lm(encoder_outputs=attn_v_reps, labels=target_encoding)
        # if is_test:
        #    print('output: ' + str(self.tokenizer.decode(torch.max(gen_caption[1][0], 1)[1])) + ' label: ' + str(
        #        self.tokenizer.decode(target_encoding[0])))
        if not self.training or True:
            list_of_caption = []
            for i in range(self.config.batch_size):
                list_of_caption.append(self.tokenizer.decode(torch.max(gen_caption[1][i], 1)[1]))
        else:
            list_of_caption = None
        # print(list_of_caption)
        # exit(1)
        return {'logits': gen_caption.loss, 'gen_caption': list_of_caption}


class GradReverse(Function):
    @staticmethod
    def forward(self, x, lambda_val):
        self.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return grad_output.neg() * self.lambda_val, None


def grad_reverse(x, lambda_val):
    return GradReverse.apply(x, lambda_val)


class DomainClassifier_A(nn.Module):
    def __init__(self, config):
        super(DomainClassifier_A, self).__init__()
        self.config = config
        self.lambda_val = self.config.lambda_val
        self.fc1 = nn.Linear(get_feat_dim_from_config(config), self.config.dc_dim)
        self.dropout = nn.Dropout(p=self.config.dc_dropout)
        classes = self.config.n_domains
        self.fc2 = nn.Linear(self.config.dc_dim, classes)

    def forward(self, x):
        x = grad_reverse(x, self.lambda_val)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return {"logits": x}

class DomainClassifier_L(nn.Module):
    def __init__(self, config):
        super(DomainClassifier_L, self).__init__()
        self.config = config
        self.lambda_val = self.config.lambda_val
        self.fc1 = nn.Linear(get_feat_dim_from_config(config), self.config.dc_dim)
        self.dropout = nn.Dropout(p=self.config.dc_dropout)
        classes = self.config.n_sources
        self.fc2 = nn.Linear(self.config.dc_dim, classes)

    def forward(self, x):
        x = grad_reverse(x, self.lambda_val)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return {"logits": x}

class LinearClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.linear = torch.nn.Linear(self.config.feat_dim, self.config.n_classes)

    def forward(self, x, text=None, target=None):
        x = x['representations'].squeeze()
        out = self.linear(x)
        return {"logits": out}


class AttentionMaskGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.softmax = torch.nn.Softmax(dim=1)

    def generate_attn(self, sim, labels, scenarios, sources, uids, attn='self'):

        if attn == "all":
            sim_sm = self.softmax(sim)

        elif attn == "other uids":
            # Set elements to 0 with no similarity. Two different methods.
            # mask = 1 - torch.eye(sim.shape[0])
            # sim_sm = self.softmax(sim) * mask
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim = sim - 1000 * torch.stack(
                [torch.where(uids == uids[i], 1, 0) for i in range(uids.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "other":
            # Set elements to 0 with no similarity. Two different methods.
            # mask = 1 - torch.eye(sim.shape[0])
            # sim_sm = self.softmax(sim) * mask
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim_sm = self.softmax(sim)
        elif attn == "other scenarios":
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 1, 0) for i in range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "other sources":

            sim = sim - 1000 * torch.stack(
                [torch.where(sources == sources[i], 1, 0) for i in range(sources.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "other scenarios other sources":
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 1, 0) for i in range(scenarios.shape[0])])
            sim = sim - 1000 * torch.stack(
                [torch.where(sources == sources[i], 1, 0) for i in range(sources.shape[0])])
            sim_sm = self.softmax(sim)

        elif attn == "other and 25":

            sim = sim - 1000 * torch.stack(
                [(torch.rand(size=(1, self.config.batch_size), device=sim.device) < 0.75).int() for i in
                 range(scenarios.shape[0])]).reshape(self.config.batch_size, self.config.batch_size)
            sim_sm = self.softmax(sim)
        elif attn == "other and 50":

            sim = sim - 1000 * torch.stack(
                [(torch.rand(size=(1, self.config.batch_size), device=sim.device) < 0.50).int() for i in
                 range(scenarios.shape[0])]).reshape(self.config.batch_size, self.config.batch_size)
            sim_sm = self.softmax(sim)
        elif attn == "other and 75":

            sim = sim - 1000 * torch.stack(
                [(torch.rand(size=(1, self.config.batch_size), device=sim.device) < 0.25).int() for i in
                 range(scenarios.shape[0])]).reshape(self.config.batch_size, self.config.batch_size)
            sim_sm = self.softmax(sim)
        elif attn == "other and same label":
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim = sim - 1000 * torch.stack(
                [torch.where(labels == labels[i], 0, 1) for i in range(labels.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "few":
            sim = sim - 1000 * torch.stack(
                [(torch.rand(size=(1, self.config.batch_size), device=sim.device) < 0.75).int() for i in
                 range(scenarios.shape[0])]).reshape(self.config.batch_size, self.config.batch_size)
            sim_sm = self.softmax(sim)
        elif attn == "one scenario":
            scenario_list = [random.randint(0, self.config.n_domains) for i in range(self.config.batch_size)]
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios != scenario_list[i], 1, 0) for i in range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "easy scenario":
            scenario_list = [3, 6, 9]
            sim = sim - 1000 * torch.stack(
                [torch.where(sum(scenarios == i for i in scenario_list).bool(), 1, 0) for i in
                 range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "medium scenario":
            scenario_list = [4, 5, 1, 7]
            sim = sim - 1000 * torch.stack(
                [torch.where(sum(scenarios == i for i in scenario_list).bool(), 1, 0) for i in
                 range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "medium scenario add":
            scenario_list = [3, 6, 9, 4, 5, 1, 7]
            sim = sim - 1000 * torch.stack(
                [torch.where(sum(scenarios == i for i in scenario_list).bool(), 1, 0) for i in
                 range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "hard scenario":
            scenario_list = [0, 2, 8]
            sim = sim - 1000 * torch.stack(
                [torch.where(sum(scenarios == i for i in scenario_list).bool(), 1, 0) for i in
                 range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "hard scenario add":
            scenario_list = [3, 6, 9, 4, 5, 1, 7, 0, 2, 8]
            sim = sim - 1000 * torch.stack(
                [torch.where(sum(scenarios == i for i in scenario_list).bool(), 1, 0) for i in
                 range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "same label other scenarios":
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 1, 0) for i in range(scenarios.shape[0])])
            sim = sim - 1000 * torch.stack(
                [torch.where(labels == labels[i], 0, 1) for i in range(labels.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "same scenario":
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 0, 1) for i in range(scenarios.shape[0])])
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim_sm = self.softmax(sim)
        elif attn == "same scenario different source":
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 0, 1) for i in range(scenarios.shape[0])])
            sim = sim - 1000 * torch.stack(
                [torch.where(sources == sources[i], 1, 0) for i in range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "same source different scenario":
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 1, 0) for i in range(scenarios.shape[0])])
            sim = sim - 1000 * torch.stack(
                [torch.where(sources == sources[i], 0, 1) for i in range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "same source":
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim = sim - 1000 * torch.stack(
                [torch.where(sources == sources[i], 0, 1) for i in range(sources.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "same scenario same source":
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 0, 1) for i in range(scenarios.shape[0])])
            sim = sim - 1000 * torch.stack(
                [torch.where(sources == sources[i], 0, 1) for i in range(sources.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "different label":
            sim = sim - 1000 * torch.stack(
                [torch.where(labels == labels[i], 0, 1) for i in range(labels.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "one scenario zero":
            scenario_list = [random.randint(0, self.config.n_domains) for i in range(self.config.batch_size)]
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios != scenario_list[i], 1, 0) for i in range(scenarios.shape[0])])
            sim_sm = self.softmax(sim) * torch.stack(
                [torch.where(scenarios != scenario_list[i], 0, 1) for i in range(scenarios.shape[0])])
        elif attn == "scenario balanced":

            same_sample = torch.eye(sim.shape[0], device=sim.device)
            same_uid = torch.stack(
                [torch.where(uids == uids[i], 1, 0) for i in range(uids.shape[0])])

            # set self and same uid to 0
            sim = sim * (1 - same_sample)
            sim = sim * (1 - same_uid)

            scenario_oh = F.one_hot(scenarios.to(torch.int64))
            domain_sums = torch.matmul(sim, scenario_oh.float())
            domain_wieghts = domain_sums / (torch.sum(domain_sums, dim=0) + 1e-8)

            scale_factors = torch.index_select(domain_wieghts, dim=1, index=scenarios)
            sim = sim * scale_factors

            # remove self
            sim = sim - 1000 * same_sample

            # remove same video
            sim = sim - 1000 * same_uid

            sim_sm = self.softmax(sim)
        return sim_sm


class ReconstructionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_dim = config.mlp_hidden_dims[-1]
        self.bottle_neck_dim = config.bottleneck_dim

        self.q_head = nn.Linear(self.model_dim, self.bottle_neck_dim)

        if self.config.sim_2 == 'v+t':
            self.k_head = nn.Linear(self.model_dim * 2, self.bottle_neck_dim)
        else:
            self.k_head = nn.Linear(self.model_dim, self.bottle_neck_dim)

        self.norm_k = nn.LayerNorm(self.bottle_neck_dim)

        self.mask_generator = AttentionMaskGenerator(config)

    def forward(self, sim1, sim2, input_for_reconstruction, labels, scenarios, sources, uids, save_flag, attn):
        queries_k = self.norm_k(self.q_head(sim1))
        supports_k = self.norm_k(self.k_head(sim2))

        # supports_v = self.v_head(supports)

        q_s_similarity = torch.matmul(queries_k, supports_k.t())  # try dividing by sqrt(self.model_dim)
        q_s_similarity = self.mask_generator.generate_attn(q_s_similarity, labels, scenarios, sources, uids, attn)
        # Drop connect implementation to avoid overfitting
        if (self.config.drop_connect and self.training):
            mask = torch.bernoulli((self.config.drop_connect_value) * torch.ones(self.config.batch_size, device=q_s_similarity.device))
            mask = mask.unsqueeze(1).expand(self.config.batch_size, self.config.batch_size)
            q_s_similarity = q_s_similarity * mask
            q_s_similarity = q_s_similarity / (q_s_similarity.sum(1, keepdim=True) + 1e-8)

        # modified from support_v = torch.matmul(q_s_similarity, supports_v)
        reconstructed_q_rep = torch.matmul(q_s_similarity, input_for_reconstruction)
        return reconstructed_q_rep, q_s_similarity


class CLIP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_ori = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.config = config
        self.softmax = torch.nn.Softmax(dim=0)
        self.dim = self.config.clip_dim

        #if config.use_clip_proj:
        #    self.proj = nn.Linear(self.dim, self.dim)

        '''
        self.text_proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim)
        )
        '''
        if self.config.clip_attn == "softmax":
            self.mask_generator = AttentionMaskGenerator(config)
        elif self.config.clip_attn == "trx":
            self.feat_reconstructor = ReconstructionAttention(self.config)

    def forward(self, video_features, text_features, uid, target=None, attn='others', sim_1_arg='v', sim_2_arg='t',
                save_flag=None,
                is_test=False):
        # normalized features

        #video_features = video_features
        # text_features = self.text_proj(text_features)

        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        uids = uid
        scenarios = target[:, 1]
        sources = target[:, 2]
        labels = target[:, 0]
        logits_per_image_video = None
        logits_per_text_video = None
        logits_per_image_text = None
        logits_per_text_text = None
        logits_per_image_ori = None
        logits_per_text_ori = None
        sim_sm = None

        if self.config.mix_video:
            if self.config.clip_attn == "softmax":

                if attn == "self":
                    video_features_weighted = video_features
                    sim_sm = torch.ones(video_features.shape[0], video_features.shape[0], device=video_features.device)
                else:
                    sim_sm = torch.matmul(video_features, video_features.t())
                    sim_sm = self.mask_generator.generate_attn(sim_sm, labels, scenarios, sources, uids, attn=attn)

                    video_features_weighted = torch.matmul(sim_sm, video_features)
            elif self.config.clip_attn == "trx":
                if attn == "self":
                    video_features_weighted = video_features
                    sim_sm = torch.ones(video_features.shape[0], video_features.shape[0], device=video_features.device)
                elif attn == "other combined":
                    sim_1 = video_features if sim_1_arg == 'v' else text_features
                    sim_2 = video_features if sim_2_arg == 'v' else text_features
                    if sim_2_arg == 'v+t':
                        sim_2 = torch.cat((video_features, text_features), dim=-1)
                    video_features_weighted, sim_sm_source = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                     labels,
                                                                                     scenarios, sources, uids,
                                                                                     save_flag,
                                                                                     attn='other sources')
                    video_features_weighted, sim_sm_scenario = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                       labels,
                                                                                       scenarios, sources, uids,
                                                                                       save_flag,
                                                                                       attn='other scenarios')
                    video_features_weighted, sim_sm_other = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                    labels,
                                                                                    scenarios, sources, uids,
                                                                                    save_flag,
                                                                                    attn='other')
                    sim = sim_sm_source + sim_sm_scenario + sim_sm_other
                    video_features_weighted = torch.matmul(sim, video_features)


                else:
                    sim_1 = video_features if sim_1_arg == 'v' else text_features
                    sim_2 = video_features if sim_2_arg == 'v' else text_features
                    if sim_2 == 'v+t':
                        sim_2 = torch.cat((video_features, text_features), dim=-1)
                    video_features_weighted, sim_sm = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                              labels,
                                                                              scenarios, sources, uids, save_flag,
                                                                              attn=attn)
                    if self.config.use_clip_proj:
                        video_features_weighted = self.proj(video_features_weighted)

            logit_scale = self.logit_scale.exp()
            logits_per_image_video = logit_scale * video_features_weighted @ text_features.float().t()
            logits_per_text_video = logits_per_image_video.t()

        if self.config.mix_txt:
            sim_text = torch.matmul(text_features, video_features.t())
            sim_text = self.mask_generator.generate_attn(sim_text, labels, scenarios)
            logit_scale_text = self.logit_scale_text.exp()
            text_features_weighted = torch.matmul(sim_text, text_features)
            logits_per_image_text = logit_scale_text * video_features @ text_features_weighted.float().t()
            logits_per_text_text = logits_per_image_text.t()

        if self.config.ori:
            logit_scale_ori = self.logit_scale_ori.exp()
            logits_per_image_ori = logit_scale_ori * video_features @ text_features.float().t()
            logits_per_text_ori = logits_per_image_ori.t()

        # shape = [global_batch_size, global_batch_size]
        return {
            'representations': {'logits_img_video': logits_per_image_video, 'logits_txt_video': logits_per_text_video,
                                'logits_img_text': logits_per_image_text, 'logits_txt_text': logits_per_text_text,
                                'logits_img_ori': logits_per_image_ori, 'logits_txt_ori': logits_per_text_ori},
            'mix_representation': video_features_weighted,
            'weights': sim_sm}


class CLIP_ori(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_ori = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.config = config
        self.softmax = torch.nn.Softmax(dim=0)
        self.dim = self.config.clip_dim

    def forward(self, video_features, text_features, target=None, attn='others', save_flag=None,
                is_test=False):
        # normalized features

        video_features = video_features
        #text_features = self.text_proj(text_features)

        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits_per_image_video = None
        logits_per_text_video = None
        logits_per_image_text = None
        logits_per_text_text = None
        logits_per_image_ori = None
        logits_per_text_ori = None

        logit_scale_ori = self.logit_scale_ori.exp()
        logits_per_image_ori = logit_scale_ori * video_features @ text_features.float().t()
        logits_per_text_ori = logits_per_image_ori.t()

        # shape = [global_batch_size, global_batch_size]
        return {
            'representations': {'logits_img_video': logits_per_image_video, 'logits_txt_video': logits_per_text_video,
                                'logits_img_text': logits_per_image_text, 'logits_txt_text': logits_per_text_text,
                                'logits_img_ori': logits_per_image_ori, 'logits_txt_ori': logits_per_text_ori},
            'weights': None}


class DoPrompt_multi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frames = get_seq_len_from_config(config)
        self.mlp = MLP(self.config, prompt_dim=self.config.prompt_dim * 2)
        self.prompt_activity_tokens = nn.Parameter(
            torch.empty(self.config.n_domains, self.config.prompt_dim).normal_(std=0.02)
        )
        self.prompt_source_tokens = nn.Parameter(
            torch.empty(self.config.n_sources, self.config.prompt_dim).normal_(std=0.02)
        )
        self.no_domain_token = nn.Parameter(
            torch.empty(1, self.config.prompt_dim * 2).normal_(std=0.02)
        )
        self.adapter_activity = nn.Sequential(
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.mlp_hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.n_domains * self.config.prompt_dim),
        )
        self.adapter_source = nn.Sequential(
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.mlp_hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.n_sources * self.config.prompt_dim),
        )

    def forward(self, x, target):
        # return: adapter_output, feat_clip, feat_cls, feat_prompt
        # standard forward pass (learned domain prompt + input)
        # forward pass for clip and adapting (no_domain prompt + input)
        # forward pass for adapted prompt (combined prompt + input)
        return_dict = {}
        domain = target[:, 1]
        source = target[:, 2]
        x = x.reshape(-1, self.config.feat_dim * self.frames)
        # standard forward pass (learned domain prompt + input)
        if self.training:
            domain_prompts = self.no_domain_token.expand(x.shape[0], -1) + torch.cat(
                (self.prompt_activity_tokens[domain.long()], self.prompt_source_tokens[source.long()]), dim=1)
            input_learned_prompt = torch.cat((x, domain_prompts), dim=1)
            feat_cls = self.mlp(input_learned_prompt)
            return_dict['feat'] = feat_cls['representations']

        # forward pass (no_domain prompt + input)
        input_no_domain_prompt = torch.cat((x, self.no_domain_token.expand(x.shape[0], -1)), dim=1)
        feat_no_prompt = self.mlp(input_no_domain_prompt)
        return_dict['feat_no_prompt'] = feat_no_prompt

        # forward pass for adapting (no_domain prompt + input)
        feat_no_prompt_s_g = feat_no_prompt['representations'].detach()
        prompt_weights_activity = self.adapter_activity(feat_no_prompt_s_g)
        prompt_weights_source = self.adapter_source(feat_no_prompt_s_g)

        prompt_weights_activity = einops.rearrange(prompt_weights_activity, 'b (d f) -> b d f',
                                                   d=self.config.n_domains,
                                                   f=self.config.prompt_dim)
        prompt_weights_source = einops.rearrange(prompt_weights_source, 'b (d f) -> b d f',
                                                 d=self.config.n_sources,
                                                 f=self.config.prompt_dim)
        prompt_weights_activity = F.softmax(prompt_weights_activity, dim=1)
        prompt_weights_source = F.softmax(prompt_weights_source, dim=1)

        return_dict['prompt_weights_source'] = prompt_weights_source
        return_dict['prompt_weights_activity'] = prompt_weights_activity

        # adapting prompt (to return combined prompt)
        comb_prompt_activity = self.prompt_activity_tokens * prompt_weights_activity
        comb_prompt_source = self.prompt_source_tokens * prompt_weights_source

        comb_prompt = torch.cat((comb_prompt_activity.sum(dim=1), comb_prompt_source.sum(dim=1)),
                                dim=1) + self.no_domain_token.expand(x.shape[0], -1)

        # forward pass for adapted prompt (combined prompt + input)
        input_adapted_prompt = torch.cat((x, comb_prompt), dim=1)
        feat_prompt = self.mlp(input_adapted_prompt)
        return_dict['feat_prompt'] = feat_prompt['representations']
        return_dict['feat_cls'] = [feat_cls['representations'],
                                   feat_prompt['representations']] if self.training else [None, feat_prompt[
            'representations']]


        return return_dict


class DoPrompt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frames = get_seq_len_from_config(config)
        self.mlp = MLP(self.config, prompt_dim=self.config.prompt_dim)
        self.prompt_tokens = nn.Parameter(
            torch.empty(self.config.n_domains, self.config.prompt_dim).normal_(std=0.02)
        )
        self.no_domain_token = nn.Parameter(
            torch.empty(1, self.config.prompt_dim).normal_(std=0.02)
        )
        self.adapter = nn.Sequential(
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.mlp_hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.n_domains * self.config.prompt_dim),
        )

    def forward(self, x, target):
        # return: adapter_output, feat_clip, feat_cls, feat_prompt
        # standard forward pass (learned domain prompt + input)
        # forward pass for clip and adapting (no_domain prompt + input)
        # forward pass for adapted prompt (combined prompt + input)
        return_dict = {}
        domain = target[:, 1]
        x = x.reshape(-1, self.config.feat_dim * self.frames)
        # standard forward pass (learned domain prompt + input)
        if self.training:
            domain_prompts = self.prompt_tokens[domain.long()] + self.no_domain_token.expand(x.shape[0], -1)
            input_learned_prompt = torch.cat((x, domain_prompts), dim=1)
            feat_cls = self.mlp(input_learned_prompt)
            return_dict['feat'] = feat_cls['representations']

        # forward pass (no_domain prompt + input)
        input_no_domain_prompt = torch.cat((x, self.no_domain_token.expand(x.shape[0], -1)), dim=1)
        feat_no_prompt = self.mlp(input_no_domain_prompt)
        return_dict['feat_no_prompt'] = feat_no_prompt

        # forward pass for adapting (no_domain prompt + input)
        feat_no_prompt_s_g = feat_no_prompt['representations'].detach()
        prompt_weights = self.adapter(feat_no_prompt_s_g)
        prompt_weights = einops.rearrange(prompt_weights, 'b (d f) -> b d f', d=self.config.n_domains,
                                          f=self.config.prompt_dim)
        prompt_weights = F.softmax(prompt_weights, dim=1)

        return_dict['prompt_weights'] = prompt_weights

        # adapting prompt (to return combined prompt)
        comb_prompt = self.prompt_tokens * prompt_weights
        comb_prompt = comb_prompt.sum(dim=1) + self.no_domain_token.expand(x.shape[0], -1)

        # forward pass for adapted prompt (combined prompt + input)
        input_adapted_prompt = torch.cat((x, comb_prompt), dim=1)
        feat_prompt = self.mlp(input_adapted_prompt)
        return_dict['feat_prompt'] = feat_prompt['representations']
        return_dict['feat_cls'] = [feat_cls['representations'],
                                   feat_prompt['representations']] if self.training else [None, feat_prompt[
            'representations']]
        return_dict['clip_before'] = feat_cls['representations'] if self.training else None
        return_dict['clip_after'] = feat_prompt['representations']
        return_dict['clip_no_prompt'] = feat_no_prompt['representations']
        return_dict['clip_adapt'] = feat_prompt['representations']

        return return_dict


class DoPrompt_source(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frames = get_seq_len_from_config(config)
        self.mlp = MLP(self.config, prompt_dim=self.config.prompt_dim)
        self.prompt_tokens = nn.Parameter(
            torch.empty(self.config.n_sources, self.config.prompt_dim).normal_(std=0.02)
        )
        self.no_domain_token = nn.Parameter(
            torch.empty(1, self.config.prompt_dim).normal_(std=0.02)
        )
        self.adapter = nn.Sequential(
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.mlp_hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.n_sources * self.config.prompt_dim),
        )

    def forward(self, x, target):
        # return: adapter_output, feat_clip, feat_cls, feat_prompt
        # standard forward pass (learned domain prompt + input)
        # forward pass for clip and adapting (no_domain prompt + input)
        # forward pass for adapted prompt (combined prompt + input)
        return_dict = {}
        domain = target[:, 2]
        x = x.reshape(-1, self.config.feat_dim * self.frames)
        # standard forward pass (learned domain prompt + input)
        if self.training:
            domain_prompts = self.prompt_tokens[domain.long()] + self.no_domain_token.expand(x.shape[0], -1)
            input_learned_prompt = torch.cat((x, domain_prompts), dim=1)
            feat_cls = self.mlp(input_learned_prompt)
            return_dict['feat'] = feat_cls['representations']

        # forward pass (no_domain prompt + input)
        input_no_domain_prompt = torch.cat((x, self.no_domain_token.expand(x.shape[0], -1)), dim=1)
        feat_no_prompt = self.mlp(input_no_domain_prompt)
        return_dict['feat_no_prompt'] = feat_no_prompt

        # forward pass for adapting (no_domain prompt + input)
        feat_no_prompt_s_g = feat_no_prompt['representations'].detach()
        prompt_weights = self.adapter(feat_no_prompt_s_g)
        prompt_weights = einops.rearrange(prompt_weights, 'b (d f) -> b d f', d=self.config.n_sources,
                                          f=self.config.prompt_dim)
        prompt_weights = F.softmax(prompt_weights, dim=1)

        return_dict['prompt_weights'] = prompt_weights

        # adapting prompt (to return combined prompt)
        comb_prompt = self.prompt_tokens * prompt_weights
        comb_prompt = comb_prompt.sum(dim=1) + self.no_domain_token.expand(x.shape[0], -1)

        # forward pass for adapted prompt (combined prompt + input)
        input_adapted_prompt = torch.cat((x, comb_prompt), dim=1)
        feat_prompt = self.mlp(input_adapted_prompt)
        return_dict['feat_prompt'] = feat_prompt['representations']
        return_dict['feat_cls'] = [feat_cls['representations'],
                                   feat_prompt['representations']] if self.training else [None, feat_prompt[
            'representations']]
        return_dict['clip_before'] = feat_cls['representations'] if self.training else None
        return_dict['clip_after'] = feat_prompt['representations']
        return_dict['clip_no_prompt'] = feat_no_prompt['representations']
        return_dict['clip_adapt'] = feat_prompt['representations']

        return return_dict


class MLP(nn.Module):
    def __init__(self, config, prompt_dim=0):
        super().__init__()
        self.config = config
        self.prompt_dim = prompt_dim
        hidden_dims = self.config.mlp_hidden_dims
        self.frames = get_seq_len_from_config(config)

        self.input_dim = config.feat_dim * self.frames + prompt_dim

        assert len(hidden_dims) >= 1
        self.bn = nn.BatchNorm1d(hidden_dims[0], affine=True)
        self.dropout = nn.Dropout(self.config.mlp_dropout)
        if self.config.reduce_dim:
            self.reduce = nn.Linear(self.input_dim, self.config.reduced_dim)
        input_mlp = self.config.reduced_dim if self.config.reduce_dim else self.input_dim
        self.input = nn.Linear((input_mlp), hidden_dims[0])

        if len(hidden_dims) > 1:
            self.hiddens = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1], affine=True),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=self.config.mlp_dropout)
                ) for i in range(len(hidden_dims) - 1)])
        else:
            self.hiddens = []

    def forward(self, x):

        x = x.reshape(self.config.batch_size, -1)

        if self.config.reduce_dim:
            x = x.reshape(-1, self.input_dim)
            x = self.reduce(x)
            x = x.reshape(self.config.batch_size, -1)

        # x = x.squeeze()
        x = self.input(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        # x = F.relu(x)
        for i, hidden in enumerate(self.hiddens):
            x = hidden(x)
            # x = self.dropout(x)
            # x = F.relu(x)

        feat = x

        # out = self.output(x)

        return {'representations': feat}
class Classifier_mix_action(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask_generator = AttentionMaskGenerator(config)
        self.feat_reconstructor = ReconstructionAttention(self.config)
        self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                self.config.n_classes)
        if self.config.use_ce_mix_proj:
            self.proj = nn.Linear(self.config.mlp_hidden_dims[-1],
                                  self.config.mlp_hidden_dims[-1])

    def forward(self, x, text, target, uid, clip_attn=None):
        if self.config.ce_norm:
            video_features = x / x.norm(dim=1, keepdim=True)
        else:
            video_features = x
        labels = target[:, 0]
        scenarios = target[:, 1]
        sources = target[:, 2]
        uids = uid
        if self.config.ce_attn == "softmax":

            sim_sm = torch.matmul(video_features, video_features.t())
            sim_sm = self.mask_generator.generate_attn(sim_sm, labels, scenarios, sources, uids, attn=self.config.attn_ce_mix)

            video_features_weighted = torch.matmul(sim_sm, video_features)
        elif self.config.ce_attn == "trx":
            if self.config.attn_ce_mix == "other combined":
                sim_1 = video_features
                sim_2 = video_features

                video_features_weighted, sim_sm_source = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                 labels,
                                                                                 scenarios, sources, uids,
                                                                                 save_flag=False,
                                                                                 attn='other sources')
                video_features_weighted, sim_sm_scenario = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                   labels,
                                                                                   scenarios, sources, uids,
                                                                                   save_flag=False,
                                                                                   attn='other scenarios')
                video_features_weighted, sim_sm_other = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                labels,
                                                                                scenarios, sources, uids,
                                                                                save_flag=False,
                                                                                attn='other')
                sim = sim_sm_source + sim_sm_scenario + sim_sm_other
                video_features_weighted = torch.matmul(sim, video_features)

            else:
                sim_1 = video_features
                sim_2 = video_features

                video_features_weighted, sim_sm = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                          labels,
                                                                          scenarios, sources, uids, save_flag=False,
                                                                          attn=self.config.attn_ce_mix)
        if self.config.use_ce_mix_proj:
            video_features_weighted = self.proj(video_features_weighted)

        out = {'logits': self.output(video_features_weighted), 'feat': video_features_weighted}
        return out

class Classifier_mix_source(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask_generator = AttentionMaskGenerator(config)
        self.feat_reconstructor = ReconstructionAttention(self.config)
        self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                self.config.n_sources)
        if self.config.use_ce_mix_proj:
            self.proj = nn.Linear(self.config.mlp_hidden_dims[-1],
                                  self.config.mlp_hidden_dims[-1])

    def forward(self, x, text, target, uid, clip_attn=None):
        if self.config.ce_norm:
            video_features = x / x.norm(dim=1, keepdim=True)
        else:
            video_features = x
        labels = target[:, 0]
        scenarios = target[:, 1]
        sources = target[:, 2]
        uids = uid
        if self.config.ce_attn == "softmax":

            sim_sm = torch.matmul(video_features, video_features.t())
            sim_sm = self.mask_generator.generate_attn(sim_sm, labels, scenarios, sources, uids, attn=self.config.attn_ce_mix)

            video_features_weighted = torch.matmul(sim_sm, video_features)
        elif self.config.ce_attn == "trx":
            if self.config.attn_ce_mix == "other combined":
                sim_1 = video_features
                sim_2 = video_features

                video_features_weighted, sim_sm_source = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                 labels,
                                                                                 scenarios, sources, uids,
                                                                                 save_flag=False,
                                                                                 attn='other sources')
                video_features_weighted, sim_sm_scenario = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                   labels,
                                                                                   scenarios, sources, uids,
                                                                                   save_flag=False,
                                                                                   attn='other scenarios')
                video_features_weighted, sim_sm_other = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                labels,
                                                                                scenarios, sources, uids,
                                                                                save_flag=False,
                                                                                attn='other')
                sim = sim_sm_source + sim_sm_scenario + sim_sm_other
                video_features_weighted = torch.matmul(sim, video_features)

            else:
                sim_1 = video_features
                sim_2 = video_features

                video_features_weighted, sim_sm = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                          labels,
                                                                          scenarios, sources, uids, save_flag=False,
                                                                          attn=self.config.attn_ce_mix)
        if self.config.use_ce_mix_proj:
            video_features_weighted = self.proj(video_features_weighted)

        out = {'logits': self.output(video_features_weighted), 'feat': video_features_weighted}
        return out

class Classifier_mix_scenario(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask_generator = AttentionMaskGenerator(config)
        self.feat_reconstructor = ReconstructionAttention(self.config)
        self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                self.config.n_domains)
        if self.config.use_ce_mix_proj:
            self.proj = nn.Linear(self.config.mlp_hidden_dims[-1],
                                  self.config.mlp_hidden_dims[-1])

    def forward(self, x, text, target, clip_attn=None):
        if self.config.ce_norm:
            video_features = x / x.norm(dim=1, keepdim=True)
        else:
            video_features = x
        labels = target[:, 0]
        scenarios = target[:, 1]
        sources = target[:, 2]
        uids = target[:, 3]
        if self.config.ce_attn == "softmax":

            sim_sm = torch.matmul(video_features, video_features.t())
            sim_sm = self.mask_generator.generate_attn(sim_sm, labels, scenarios, sources, uids, attn=self.config.attn_ce_mix)

            video_features_weighted = torch.matmul(sim_sm, video_features)
        elif self.config.ce_attn == "trx":
            if self.config.attn_ce_mix == "other combined":
                sim_1 = video_features
                sim_2 = video_features

                video_features_weighted, sim_sm_source = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                 labels,
                                                                                 scenarios, sources, uids,
                                                                                 save_flag=False,
                                                                                 attn='other sources')
                video_features_weighted, sim_sm_scenario = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                   labels,
                                                                                   scenarios, sources, uids,
                                                                                   save_flag=False,
                                                                                   attn='other scenarios')
                video_features_weighted, sim_sm_other = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                                labels,
                                                                                scenarios, sources, uids,
                                                                                save_flag=False,
                                                                                attn='other')
                sim = sim_sm_source + sim_sm_scenario + sim_sm_other
                video_features_weighted = torch.matmul(sim, video_features)

            else:
                sim_1 = video_features
                sim_2 = video_features

                video_features_weighted, sim_sm = self.feat_reconstructor(sim_1, sim_2, video_features,
                                                                          labels,
                                                                          scenarios, sources, uids, save_flag=False,
                                                                          attn=self.config.attn_ce_mix)
        if self.config.use_ce_mix_proj:
            video_features_weighted = self.proj(video_features_weighted)

        out = {'logits': self.output(video_features_weighted), 'feat': video_features_weighted}
        return out

class Classifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if (self.config.model == 'MLP') or (self.config.model == 'DoPrompt'):
            self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                    self.config.n_classes)
        elif self.config.model == 'MLP_expert':
            self.output = nn.Linear(self.config.mlp_hidden_dims[-1] * 2,
                                    self.config.n_classes)

        if self.config.model == 'Transformer':
            self.output = torch.nn.Linear(self.config.tfmr_d_model, self.config.n_classes)

        if self.config.model == 'TRN':
            self.output = nn.ModuleList()
            for i in range(self.config.num_scales):
                classifier = nn.Linear(self.config.trn_bottleneck, self.config.n_classes)
                self.output += [classifier]

    def forward(self, x, x_2=None):
        # x = torch.cat((x, text), dim=1)
        if x_2 != None:
            out = {'logits': self.output(x), 'logits_2': self.output(x_2)}
        else:
            out = {'logits': self.output(x)}
        return out


class D_S_Classifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if (self.config.model == 'MLP') or (self.config.model == 'DoPrompt'):
            self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                    self.config.n_domains * self.config.n_sources)

        if self.config.model == 'Transformer':
            self.output = torch.nn.Linear(self.config.tfmr_d_model, self.config.n_classes)

        if self.config.model == 'TRN':
            self.output = nn.ModuleList()
            for i in range(self.config.num_scales):
                classifier = nn.Linear(self.config.trn_bottleneck, self.config.n_classes)
                self.output += [classifier]

    def forward(self, x):
        # x = torch.cat((x, text), dim=1)
        out = {'logits': self.output(x)}
        return out


class D_Classifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if (self.config.model == 'MLP') or (self.config.model == 'DoPrompt'):
            self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                    self.config.n_domains)

        if self.config.model == 'Transformer':
            self.output = torch.nn.Linear(self.config.tfmr_d_model, self.config.n_classes)

        if self.config.model == 'TRN':
            self.output = nn.ModuleList()
            for i in range(self.config.num_scales):
                classifier = nn.Linear(self.config.trn_bottleneck, self.config.n_classes)
                self.output += [classifier]

    def forward(self, x):
        # x = torch.cat((x, text), dim=1)
        out = {'logits': self.output(x)}
        return out


class S_Classifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if (self.config.model == 'MLP') or (self.config.model == 'DoPrompt'):
            self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                    self.config.n_sources)

        if self.config.model == 'Transformer':
            self.output = torch.nn.Linear(self.config.tfmr_d_model, self.config.n_classes)

        if self.config.model == 'TRN':
            self.output = nn.ModuleList()
            for i in range(self.config.num_scales):
                classifier = nn.Linear(self.config.trn_bottleneck, self.config.n_classes)
                self.output += [classifier]

    def forward(self, x):
        # x = torch.cat((x, text), dim=1)
        out = {'logits': self.output(x)}
        return out


class DoPrompt_Classifier(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if (self.config.model == 'MLP') or (self.config.model == 'DoPrompt'):
            self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                    self.config.n_classes)

        if self.config.model == 'Transformer':
            self.output = torch.nn.Linear(self.config.tfmr_d_model, self.config.n_classes)

        if self.config.model == 'TRN':
            self.output = nn.ModuleList()
            for i in range(self.config.num_scales):
                classifier = nn.Linear(self.config.trn_bottleneck, self.config.n_classes)
                self.output += [classifier]

    def forward(self, x):
        if self.training:
            x_cls = x[0]
            out_cls = self.output(x_cls)
        else:
            out_cls = None
        if x[1] != None:
            x_prompt = x[1]
            out_prompt = self.output(x_prompt)
        out = {'out_cls': out_cls, 'out_prompt': out_prompt}
        return out


class RelationModuleMultiScale(torch.nn.Module):
    # relation module in multi-scale with a classifier at the end
    def __init__(self, config):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = config.trn_scales  # how many relations selected to sum up
        self.img_feature_dim = config.feat_dim
        self.use_classifier = config.trn_classifier
        self.add_fc = config.add_fc
        self.fc_dim = config.fc_dim
        self.num_frames = get_seq_len_from_config(config) if not config.stride_1 else 5
        self.scales = [i for i in
                       range(self.num_frames, 1, -1)]  #

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(self.num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num,
                                             len(relations_scale)))  # how many samples of relation to select in each forward pass

        self.num_class = config.n_classes
        self.num_bottleneck = config.trn_bottleneck
        self.fc_fusion_scales = nn.ModuleList()  # high-tech modulelist
        # self.classifier_scales = nn.ModuleList()
        if self.add_fc:
            self.new_fc = nn.Sequential(
                nn.Linear(self.img_feature_dim, self.fc_dim),
                nn.ReLU(),
                nn.Dropout(p=config.fc_dropout)  # this is the newly added thing
            )
        for i in range(len(self.scales)):
            scale = self.scales[i]

            fc_fusion = nn.Sequential(
                nn.ReLU(),
                nn.Linear(scale * self.img_feature_dim, self.num_bottleneck),
                nn.ReLU(),
                nn.Dropout(p=config.trn_dropout),  # this is the newly added thing
                nn.Linear(self.num_bottleneck, self.num_bottleneck),
                nn.ReLU(),
                nn.Dropout(p=config.trn_dropout),
            )
            self.fc_fusion_scales += [fc_fusion]

            # if self.use_classifier:
            #    classifier = nn.Linear(self.num_bottleneck, self.num_class)
            #    self.classifier_scales += [classifier]
        # maybe we put another fc layer after the summed up results???
        print('Multi-Scale Temporal Relation with classifier in use')
        print(['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale

        if self.add_fc:
            self.new_fc(input)
        act_all = input[:, self.relations_scales[0][0], :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)
        act_all = self.classifier_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]),
                                                          self.subsample_scales[scaleID], replace=False)
            for i, idx in enumerate(idx_relations_randomsample):
                act_relation = input[:, self.relations_scales[scaleID][idx], :]

                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                if i == 0:
                    feat = act_relation.unsqueeze(1)
                else:

                    torch.cat((feat, act_relation.unsqueeze(1)), dim=1)
                # if self.use_classifier:
                #   act_relation = self.classifier_scales[scaleID](act_relation)
                # act_all += act_relation
        mean_feat = feat.mean(dim=1).squeeze(1)
        # feat = feat.reshape(-1, self.num_bottleneck)
        # dictionary, and then an if on whether the dict contains multi-scale feat
        return {'representations': mean_feat, 'scale_representations': feat}
        # return {"logits": act_all, "representations": feat}

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


# Adapted from MTCN
class Transformer(nn.Module):
    """
    Reduces dimensionality of feature input.
    Appends a learned CLS token.
    Adds learned positional embeddings.
    Passes though a transformer encoder.
    Linear layer from output of CLS token gives logits.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(self.config.feat_dim, self.config.tfmr_d_model)
        self.input_projection_relu = nn.ReLU()
        if self.config.use_domain_prompt:
            pos_dim = get_seq_len_from_config(config) + 5 if self.config.learn_prompt else get_seq_len_from_config(
                config) + 2
        else:
            pos_dim = get_seq_len_from_config(config) + 1
        self.positional_embedding = nn.Parameter(
            torch.empty((1, pos_dim, self.config.tfmr_d_model), requires_grad=True))

        normal_(self.positional_embedding, std=0.001)
        # prompt tokens
        if self.config.use_domain_prompt:
            self.prompt_tokens = nn.Parameter(
                torch.empty(10, 4, self.config.tfmr_d_model).normal_(std=0.02)
            )
        self.CLS_token = nn.Parameter(torch.empty((1, 1, self.config.tfmr_d_model), requires_grad=True))
        normal_(self.CLS_token, std=0.001)

        encoder_layer = TransformerEncoderLayer(d_model=self.config.tfmr_d_model,
                                                nhead=self.config.tfmr_n_heads,
                                                dim_feedforward=self.config.tfmr_d_feedforward,
                                                dropout=self.config.tfmr_dropout)
        if self.config.tfmr_shared:
            self.transformer_encoder = ParameterSharedTransformerEncoder(encoder_layer, num_unique_layers=1,
                                                                         num_total_layers=self.config.tfmr_n_layers,
                                                                         mode='sequence')
        else:
            self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=self.config.tfmr_n_layers)

        self.classification_linear = torch.nn.Linear(self.config.tfmr_d_model, self.config.n_classes)

        # TODO dropout? on features

    def forward(self, x):
        x_red = self.input_projection(x)
        x_red = self.input_projection_relu(x_red)
        CLS_token = self.CLS_token.expand(x_red.shape[0], -1, -1)
        x_red = torch.cat([CLS_token, x_red], dim=1)

        # add positional encoding
        x_red_pos = x_red + self.positional_embedding

        # pass through transformer. Transformer is seq_len x batch x dim
        tfmr_in = einops.rearrange(x_red_pos, 'b s d -> s b d')
        tfmr_out = self.transformer_encoder(tfmr_in)

        # put back into batch x seq_len x d and get transformed CLS token
        tfmr_out = einops.rearrange(tfmr_out, 's b d -> b s d')
        CLS_out = tfmr_out[:, 0, :]

        # out = self.classification_linear(CLS_out)
        feat = tfmr_out[:, 1:, :].mean(dim=1).squeeze(1)
        return {'representations': feat, 'cls_out': CLS_out}


class ParameterSharedTransformerEncoder(nn.TransformerEncoder):
    def __init__(
            self,
            encoder_layer,
            num_unique_layers=3,
            num_total_layers=6,
            mode="cycle_rev",
            norm=False,
    ):
        assert mode in {"sequence", "cycle", "cycle_rev"}
        quotient, remainder = divmod(num_total_layers, num_unique_layers)
        assert remainder == 0
        if mode == "cycle_rev":
            assert quotient == 2
        encoder_layer = encoder_layer
        super().__init__(encoder_layer, num_layers=num_unique_layers, norm=norm)
        self.N = num_total_layers
        self.M = num_unique_layers
        self.mode = mode
        # self.norm = nn.LayerNorm(d_model) if norm else None

    def forward(self, x, mask=None, src_key_padding_mask=None, verbose=False):
        for i in range(self.N):
            if self.mode == "sequence":
                i = i // (self.N // self.M)
            elif self.mode == "cycle":
                i = i % self.M
            elif i > (self.N - 1) / 2:
                i = self.N - i - 1
            if verbose:
                print(f"layer {i}")
            x = self.layers[i](x, mask, src_key_padding_mask)
        # if self.norm is not None:
        #    x = self.norm(x)
        return x


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        # clip_model=clip_model.half()
        # convert_weights(clip_model)
        self.transformer = clip_model.transformer  # .half()
        # convert_weights(self.transformer)
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.r = torch.arange(300).to('cuda')
        print(self.dtype)

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        s = time.time()
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[self.r, tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CustomCLIP(nn.Module):
    def __init__(self, config, clip_model):
        super().__init__()
        from utils import verb_dict
        self.prompt_learner = PromptLearner(config, verb_dict.keys(), clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image_feat, text_feat, label, is_test=False):
        tokenized_prompts = self.tokenized_prompts
        image_feat = image_feat['representations']
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feat.float() @ text_features.float().t()

        return logits


class PromptLearner(nn.Module):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.config = config
        n_ctx = 8
        ctx_init = ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)

            ctx_init_d = ctx_init.replace("_", " ")
            n_ctx_d = len(ctx_init_d.split(" "))
            prompt_d = clip.tokenize(ctx_init_d)

            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
                embedding_d = clip_model.token_embedding(prompt_d).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            ctx_vectors_d = embedding_d[0, 1: 1 + n_ctx_d, :]
            prompt_prefix = ctx_init
            prompt_d_prefix = ctx_init_d
        else:
            # random initialization
            if self.config.generic_context:
                ctx_vectors = torch.empty(self.n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_d = torch.empty(self.n_cls, 10, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                nn.init.normal_(ctx_vectors_d, std=0.02)
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_d = torch.empty(10, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                nn.init.normal_(ctx_vectors_d, std=0.02)

            prompt_prefix = " ".join(["X"] * n_ctx)
            prompt_d_prefix = [" ".join(["X"] * n_ctx)] * 10

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Domain context: "{prompt_d_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx_d = nn.Parameter(ctx_vectors_d)
        classnames = [name.replace("-", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts = [prompt_prefix + " " + prompt_d_prefix[i] + " " + name + "." for name in classnames for i in
                   range(10)]
        # (60, 10, 512)
        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in prompts]).to(
            torch.device('cuda'))  # .to(torch.device('cuda'))  # (n_cls*n_scenarios, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx * 2:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        ctx_d = self.ctx_d

        with open('token_new.txt', 'w') as f:
            f.write('ctx: \n' + str(ctx))
            f.write('ctx_d: \n' + str(ctx_d))

        if ctx_d.dim() == 3:
            ctx_d = ctx_d.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        # for i in range(len(ctx_d)):
        #    if ctx_d[i].dim() == 2:
        #       ctx_d[i] =  nn.Parameter(ctx_d[i].unsqueeze(0).expand(self.n_cls, -1, -1))

        # ctx_d =  torch.stack(ctx_d_p, dim=0)
        ctx_d = ctx_d.reshape(-1, ctx_d.size()[-2], ctx_d.size()[-1])
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls * 10, -1, -1)
        else:
            ctx = ctx.unsqueeze(0).expand(10, -1, -1, -1)
            ctx = ctx.reshape(-1, ctx.size()[-2], ctx.size()[-1])
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.class_token_position == "end":
            # [(n_cls, n_ctx*2+1+suffix, dim)]*9
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    ctx_d,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            # [(n_cls*10, n_ctx*2+1+suffix, dim)]

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class Mixup(nn.Module):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config

    def __call__(self, x, target):
        objective = 0
        target = target[:, 0]
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = target, target[index]

        return {'representations': mixed_x if self.training else x, 'y_a': y_a, 'y_b': y_b, 'lam': lam}
