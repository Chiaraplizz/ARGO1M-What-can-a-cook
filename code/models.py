import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from utils import get_seq_len_from_config, get_feat_dim_from_config
from torch.autograd import Function
from sentence_transformers import SentenceTransformer, util

"""
Code for implementing video and text projectors, CIR-text and CIR modules, and baseline methods (DoPrompt, Mixup and DANN). 
"""


class TextModel(nn.Module):
    """
        TextModel computes text embedding using the SentenceTransformer library.
        It takes in a text input, encodes it using a pre-trained model, and performs a projection on top of it.
    """

    def __init__(self, config):
        super().__init__()
        self.model = SentenceTransformer('sentence-transformers/' + config.text_model)
        self.text_dim = config.text_dim
        self.dim = config.CIR_text_dim
        self.config = config
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim)
        )

    def forward(self, text):
        return {'text_feat': self.text_proj(self.model.encode(text, convert_to_tensor=True))}


class MLP(nn.Module):
    """
        MLP module for computing video embeddings.
    """

    def __init__(self, config, prompt_dim=0):
        super().__init__()
        self.config = config
        self.prompt_dim = prompt_dim
        hidden_dims = self.config.mlp_hidden_dims
        self.frames = get_seq_len_from_config(config)
        self.input_dim = config.feat_dim * self.frames + self.prompt_dim
        assert len(hidden_dims) >= 1
        self.bn = nn.BatchNorm1d(hidden_dims[0], affine=True)
        self.dropout = nn.Dropout(self.config.mlp_dropout)
        self.input = nn.Linear((self.input_dim), hidden_dims[0])

        self.hiddens = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1], affine=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=self.config.mlp_dropout)
            ) for i in range(len(hidden_dims) - 1)])

    def forward(self, x):
        x = x.reshape(self.config.batch_size, -1)
        x = self.input(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        for i, hidden in enumerate(self.hiddens):
            x = hidden(x)
        return {'representations': x}


class Classifier(torch.nn.Module):
    """
         Module for action classification
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                self.config.n_classes)

    def forward(self, x, x_mixed=None):

        if x_mixed != None and self.training:
            out = {'logits': self.output(x), 'logits_mixed': self.output(x_mixed)}
        else:
            out = {'logits': self.output(x)}
        return out


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


class DomainClassifier(nn.Module):
    """
        Module for domain classification. A Gradient Reversal Layer (GRL) is added before that to revert the gradient.
    """

    def __init__(self, config, index):
        super(DomainClassifier, self).__init__()
        self.config = config
        self.index = index
        self.lambda_val = self.config.lambda_val
        self.fc1 = nn.Linear(get_feat_dim_from_config(config), self.config.dc_dim)
        self.dropout = nn.Dropout(p=self.config.dc_dropout)
        classes = self.config.n_scenarios if self.index == 1 else self.config.n_locations
        self.fc2 = nn.Linear(self.config.dc_dim, classes)

    def forward(self, x):
        x = grad_reverse(x, self.lambda_val)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return {"logits": x}


class DomainClassifier_S(DomainClassifier):
    """
        Module for scenario classification.
    """

    def __init__(self, config):
        super(DomainClassifier_S, self).__init__(config, index=1)


class DomainClassifier_L(DomainClassifier):
    """
        Module for location classification.
    """

    def __init__(self, config):
        super(DomainClassifier_L, self).__init__(config, index=2)


class AttentionMaskGenerator(nn.Module):
    """
    Generates attention mask based on different combinations of scenarios and locations in the batch.

    Args:
        sim (torch.Tensor): Similarity matrix.
        scenarios (torch.Tensor): Tensor containing scenario information.
        locations (torch.Tensor): Tensor containing location information.
        attn (str): Type of attention mask to generate. Possible values are "all", "other", "OS",
            "OL", "OS-OL", "SS", "SL", "SS-SL". (OS: Other Scenario, OL: Other Location,
            SS: Same Scenario, SL: Same Location, SS-SL: Same Scenario - Same Location)

    Returns:
        torch.Tensor: Softmax attention mask.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.softmax = torch.nn.Softmax(dim=1)

    def generate_attn(self, sim, scenarios, locations, attn='other'):
        if attn == "all":
            sim_sm = self.softmax(sim)
        elif attn == "other":
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim_sm = self.softmax(sim)
        elif attn == "OS":
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 1, 0) for i in range(scenarios.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "OL":
            sim = sim - 1000 * torch.stack(
                [torch.where(locations == locations[i], 1, 0) for i in range(locations.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "OS-OL":
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 1, 0) for i in range(scenarios.shape[0])])
            sim = sim - 1000 * torch.stack(
                [torch.where(locations == locations[i], 1, 0) for i in range(locations.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "SS":
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 0, 1) for i in range(scenarios.shape[0])])
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim_sm = self.softmax(sim)
        elif attn == "SL":
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim = sim - 1000 * torch.stack(
                [torch.where(locations == locations[i], 0, 1) for i in range(locations.shape[0])])
            sim_sm = self.softmax(sim)
        elif attn == "SS-SL":
            sim = sim - 1000 * torch.eye(sim.shape[0], device=sim.device)
            sim = sim - 1000 * torch.stack(
                [torch.where(scenarios == scenarios[i], 0, 1) for i in range(scenarios.shape[0])])
            sim = sim - 1000 * torch.stack(
                [torch.where(locations == locations[i], 0, 1) for i in range(locations.shape[0])])
            sim_sm = self.softmax(sim)
        return sim_sm


class ReconstructionAttention(nn.Module):
    """
         Learnable attention.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_dim = config.mlp_hidden_dims[-1]
        self.bottle_neck_dim = config.bottleneck_dim

        self.q_head = nn.Linear(self.model_dim, self.bottle_neck_dim)

        self.k_head = nn.Linear(self.model_dim, self.bottle_neck_dim)

        self.norm_k = nn.LayerNorm(self.bottle_neck_dim)

        self.mask_generator = AttentionMaskGenerator(config)

    def forward(self, input_for_reconstruction, scenarios, locations, attn):
        queries_k = self.norm_k(self.q_head(input_for_reconstruction))
        supports_k = self.norm_k(self.k_head(input_for_reconstruction))

        q_s_similarity = torch.matmul(queries_k, supports_k.t())
        q_s_similarity = self.mask_generator.generate_attn(q_s_similarity, scenarios, locations, attn)

        reconstructed_q_rep = torch.matmul((q_s_similarity), input_for_reconstruction)
        return reconstructed_q_rep, q_s_similarity


class CIR_text(nn.Module):
    """
        CIR-text reconstruction module.
    """

    def __init__(self, config):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.config = config
        self.softmax = torch.nn.Softmax(dim=0)
        self.dim = self.config.CIR_text_dim

        if self.config.CIR_text_attn == "fixed":
            self.mask_generator = AttentionMaskGenerator(self.config)
        elif self.config.CIR_text_attn == "learned":
            self.feat_reconstructor = ReconstructionAttention(self.config)

    def forward(self, video_features, text_features=None, target=None):

        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        labels = target[:, 0]
        scenarios = target[:, 1]
        locations = target[:, 2]

        sim_sm = None

        if self.config.CIR_text_attn == "fixed":
            sim_sm = torch.matmul(video_features, video_features.t())
            sim_sm = self.mask_generator.generate_attn(sim_sm, labels, scenarios, locations, attn=self.config.gen_attn)
            video_features_weighted = torch.matmul(sim_sm, video_features)
        elif self.config.CIR_text_attn == "learned":
            video_features_weighted, sim_sm = self.feat_reconstructor(video_features,
                                                                      scenarios, locations,
                                                                      attn=self.config.gen_attn)
        logit_scale = self.logit_scale.exp()
        logits_per_video = logit_scale * video_features_weighted @ text_features.float().t()
        logits_per_text = logits_per_video.t()

        return {
            'representations': {'logits_video': logits_per_video, 'logits_txt': logits_per_text},
            'mix_representation': video_features_weighted,
            'weights': sim_sm}


class CIR(torch.nn.Module):
    """
        CIR-action reconstruction module.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mask_generator = AttentionMaskGenerator(config)
        self.feat_reconstructor = ReconstructionAttention(self.config)

    def forward(self, x, target=None):

        scenarios = target[:, 1]
        locations = target[:, 2]
        if self.config.CIR_attn == "fixed":
            sim_sm = torch.matmul(x, x.t())
            sim_sm = self.mask_generator.generate_attn(sim_sm, scenarios, locations,
                                                       attn=self.config.gen_attn)
            video_features_weighted = torch.matmul(sim_sm, x)
        elif self.config.CIR_attn == "learned":
            video_features_weighted, sim_sm = self.feat_reconstructor(x,
                                                                      scenarios, locations,
                                                                      attn=self.config.gen_attn)
        else:
            video_features_weighted = x
            sim_sm = None

        out = {'feat': video_features_weighted, 'weights': sim_sm}
        return out


class DoPrompt(nn.Module):
    """
         Module implementing DoPrompt baseline method.

    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.frames = get_seq_len_from_config(config)
        self.mlp = MLP(self.config, prompt_dim=self.config.prompt_dim * 2)
        self.prompt_scenario_tokens = nn.Parameter(
            torch.empty(self.config.n_scenarios, self.config.prompt_dim).normal_(std=0.02)
        )
        self.prompt_location_tokens = nn.Parameter(
            torch.empty(self.config.n_locations, self.config.prompt_dim).normal_(std=0.02)
        )
        self.no_domain_token = nn.Parameter(
            torch.empty(1, self.config.prompt_dim * 2).normal_(std=0.02)
        )
        self.adapter_scenario = nn.Sequential(
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.mlp_hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.n_scenarios * self.config.prompt_dim),
        )
        self.adapter_location = nn.Sequential(
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.mlp_hidden_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.mlp_hidden_dims[-1], self.config.n_locations * self.config.prompt_dim),
        )

    def forward(self, x, target):
        # return: adapter_output, feat_clip, feat_cls, feat_prompt
        # standard forward pass (learned domain prompt + input)
        # forward pass for clip and adapting (no_domain prompt + input)
        # forward pass for adapted prompt (combined prompt + input)
        scenario = target[:, 1]
        location = target[:, 2]
        return_dict = {}
        x = x.reshape(-1, self.config.feat_dim * self.frames)
        # standard forward pass (learned domain prompt + input)
        if self.training:
            domain_prompts = self.no_domain_token.expand(x.shape[0], -1) + torch.cat(
                (self.prompt_scenario_tokens[scenario.long()], self.prompt_location_tokens[location.long()]), dim=1)
            input_learned_prompt = torch.cat((x, domain_prompts), dim=1)
            feat_cls = self.mlp(input_learned_prompt)
            return_dict['feat'] = feat_cls['representations']

        # forward pass (no_domain prompt + input)
        input_no_domain_prompt = torch.cat((x, self.no_domain_token.expand(x.shape[0], -1)), dim=1)
        feat_no_prompt = self.mlp(input_no_domain_prompt)
        return_dict['feat_no_prompt'] = feat_no_prompt

        # forward pass for adapting (no_domain prompt + input)
        feat_no_prompt_s_g = feat_no_prompt['representations'].detach()
        prompt_weights_scenario = self.adapter_scenario(feat_no_prompt_s_g)
        prompt_weights_location = self.adapter_location(feat_no_prompt_s_g)

        prompt_weights_scenario = einops.rearrange(prompt_weights_scenario, 'b (d f) -> b d f',
                                                   d=self.config.n_scenarios,
                                                   f=self.config.prompt_dim)
        prompt_weights_location = einops.rearrange(prompt_weights_location, 'b (d f) -> b d f',
                                                   d=self.config.n_locations,
                                                   f=self.config.prompt_dim)
        prompt_weights_scenario = F.softmax(prompt_weights_scenario, dim=1)
        prompt_weights_location = F.softmax(prompt_weights_location, dim=1)

        return_dict['prompt_weights_scenario'] = prompt_weights_scenario
        return_dict['prompt_weights_location'] = prompt_weights_location

        # adapting prompt (to return combined prompt)
        comb_prompt_scenario = self.prompt_scenario_tokens * prompt_weights_scenario
        comb_prompt_location = self.prompt_location_tokens * prompt_weights_location

        comb_prompt = torch.cat((comb_prompt_scenario.sum(dim=1), comb_prompt_location.sum(dim=1)),
                                dim=1) + self.no_domain_token.expand(x.shape[0], -1)

        # forward pass for adapted prompt (combined prompt + input)
        input_adapted_prompt = torch.cat((x, comb_prompt), dim=1)
        feat_prompt = self.mlp(input_adapted_prompt)
        return_dict['feat_prompt'] = feat_prompt['representations']
        return_dict['feat_cls'] = [feat_cls['representations'],
                                   feat_prompt['representations']] if self.training else [None, feat_prompt[
            'representations']]

        return return_dict


class DoPrompt_Classifier(torch.nn.Module):
    """
         DoPrompt classifier.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(self.config.mlp_hidden_dims[-1],
                                self.config.n_classes)

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
        target = target[:, 0]
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = target, target[index]

        return {'representations': mixed_x, 'y_a': y_a, 'y_b': y_b, 'lam': lam}
