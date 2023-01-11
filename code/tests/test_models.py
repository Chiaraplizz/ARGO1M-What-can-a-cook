from models import MLP, Transformer, RelationModuleMultiScale, GenerativeCaption, CLIP, ReconstructionAttention, DoPrompt, DoPrompt_multi
import torch
import torch.nn as nn
import numpy as np
import pytest
from utils import get_seq_len_from_config


def test_mlp():
    class DumbConfig(object):
        pass
    config = DumbConfig()
    config.feat_dim = 2304
    config.model = "mlp"
    config.mlp_hidden_dims = [4096, 4096]
    config.mlp_dropout = 0.1
    config.n_classes = 25
    config.batch_size = 5
    config.n_action_subsample = 1
    config.n_before_after_context = 0
    config.reduce_dim = True
    config.reduced_dim = 512

    data = torch.rand(config.batch_size, get_seq_len_from_config(config), config.feat_dim)
    correct_ouput_shape = torch.rand(config.batch_size, config.mlp_hidden_dims[-1]).shape

    model = MLP(config)
    output = model(data)

    assert output['representations'].shape == correct_ouput_shape


def test_DoPrompt():
    class DumbConfig(object):
        pass
    print('start testing')
    config = DumbConfig()
    config.prompt_dim = 5
    config.n_domains = 2
    config.feat_dim = 8
    config.model = "mlp"
    config.mlp_hidden_dims = [16, 16]
    config.mlp_dropout = 0.1
    config.n_classes = 25
    config.batch_size = 5
    config.n_action_subsample = 1
    config.n_before_after_context = 1
    config.reduce_dim = True
    config.reduced_dim = 512

    data = torch.rand(config.batch_size, get_seq_len_from_config(config), config.feat_dim)
    correct_ouput_shape = torch.rand(config.batch_size, config.n_classes).shape
    target_class = torch.randint(low=1, high=config.n_classes, size=[config.batch_size, 1])
    target_dom = torch.randint(low=0, high=config.n_domains - 1, size=[config.batch_size, 1])
    target = torch.cat([target_class, target_dom], dim=-1)

    model = DoPrompt(config)
    output = model(data, text=None, target=target, save_flag=None, is_test=False)
    assert output['prompt_weights'].shape == torch.rand(config.batch_size, config.n_domains, config.prompt_dim).shape
    assert output['feat_cls']['representations'].shape == torch.rand(config.batch_size, config.mlp_hidden_dims[-1]).shape
    assert output['feat_clip']['representations'].shape == torch.rand(config.batch_size, config.mlp_hidden_dims[-1]).shape
    assert output['feat_prompt']['representations'].shape == torch.rand(config.batch_size, config.mlp_hidden_dims[-1]).shape

def test_DoPrompt_multi():
    class DumbConfig(object):
        pass
    print('start testing')
    config = DumbConfig()
    config.prompt_dim = 5
    config.n_domains = 2
    config.n_sources = 3
    config.feat_dim = 8
    config.model = "mlp"
    config.mlp_hidden_dims = [16, 16]
    config.mlp_dropout = 0.1
    config.n_classes = 25
    config.batch_size = 5
    config.n_action_subsample = 1
    config.n_before_after_context = 1
    config.reduce_dim = True
    config.reduced_dim = 512

    data = torch.rand(config.batch_size, get_seq_len_from_config(config), config.feat_dim)
    correct_ouput_shape = torch.rand(config.batch_size, config.n_classes).shape
    target_class = torch.randint(low=1, high=config.n_classes, size=[config.batch_size, 1])
    target_dom = torch.randint(low=0, high=config.n_domains - 1, size=[config.batch_size, 1])
    target_source = torch.randint(low=0, high=config.n_sources - 1, size=[config.batch_size, 1])

    target = torch.cat([target_class, target_dom, target_source], dim=-1)

    model = DoPrompt_multi(config)
    output = model(data, target=target)


def test_TRN():
    class DumbConfig(object):
        pass

    config = DumbConfig()

    config.feat_dim = 2304
    config.model = "RelationModuleMultiScale"
    config.n_classes = 25
    config.batch_size = 5
    config.n_action_subsample = 3
    config.n_before_after_context = 2
    config.trn_scales = 2
    config.trn_classifier = True
    config.trn_bottleneck = 256
    config.trn_dropout = 0.5
    config.add_fc = 0.5
    config.fc_dim = 512
    config.fc_dropout = 0.5

    data = torch.rand(config.batch_size, get_seq_len_from_config(config), config.feat_dim)
    correct_ouput_shape = torch.rand(config.batch_size, config.n_classes).shape

    model = RelationModuleMultiScale(config)
    print(data.shape)
    output = model(data)
    print(output.shape)
    assert output.shape == correct_ouput_shape


def test_transformer():
    class DumbConfig(object):
        pass

    config = DumbConfig()

    config.feat_dim = 2304
    config.n_classes = 25
    config.batch_size = 5
    config.n_action_subsample = 3
    config.n_before_after_context = 2

    config.model = 'Transformer'

    config.tfmr_d_model = 512
    config.tfmr_n_layers = 2
    config.tfmr_n_heads = 4
    config.tfmr_d_feedforward = 1024
    config.tfmr_dropout = 0.5

    data = torch.rand(config.batch_size, get_seq_len_from_config(config), config.feat_dim)
    correct_ouput_shape = torch.rand(config.batch_size, config.n_classes).shape

    model = Transformer(config)
    output = model(data)

    assert output.shape == correct_ouput_shape


def test_CLIP():
    class DumbConfig(object):
        pass

    config = DumbConfig()

    config.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    config.logit_scale_text = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    config.logit_scale_ori = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    config.softmax = torch.nn.Softmax(dim=0)
    config.clip_dim = 512
    config.n_classes = 60
    config.n_domains = 10
    config.batch_size = 1
    config.mix_txt = False
    config.ori = True
    config.gen_attn = 'other scenarios'
    config.text_proj = nn.Sequential(
        nn.Linear(config.clip_dim, config.clip_dim),
        nn.ReLU(True),
        nn.Linear(config.clip_dim, config.clip_dim)
    )
    target_class = torch.randint(low=1, high=config.n_classes, size=[config.batch_size, 1])
    target_dom = torch.randint(low=0, high=config.n_domains - 1, size=[config.batch_size, 1])
    target = torch.cat([target_class, target_dom], dim=-1)
    config.batch_size = 1
    data_video = {}
    data_text = {}
    data_video['representations'] = torch.rand(config.batch_size, config.clip_dim)
    data_text['verb'] = torch.rand(config.batch_size, config.clip_dim)

    model = CLIP(config)
    output = model(data_video, data_text, target)

def test_ReconstructionAttention():
    class DumbConfig(object):
        pass

    config = DumbConfig()
    config.reduced_dim = 512
    config.batch_size = 8
    config.gen_attn = "other"

    queries = torch.rand(config.batch_size, config.reduced_dim)
    # supports = torch.rand(config.batch_size, config.reduced_dim)
    supports = queries
    labels = torch.tensor([0,1,2,3,0,1,2,3])
    scenarios = torch.tensor([0,1,1,0,0,1,0,0])

    reconstruction_attn = ReconstructionAttention(config)

    output = reconstruction_attn(queries, supports, labels, scenarios)
    assert(output.shape == queries.shape)





