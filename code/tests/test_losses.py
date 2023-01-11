import torch
from losses import *


def test_GenerativeCaption():
    class DumbConfig(object):
        pass

    config = DumbConfig()
    config.feat_dim = 2304
    config.model = "gen"
    config.mlp_hidden_dims = [512]
    config.mlp_dropout = 0.1
    config.n_classes = 25
    config.batch_size = 5
    config.n_action_subsample = 1
    config.n_before_after_context = 0
    config.reduce_dim = True
    config.reduced_dim = 512
    config.clip_dim = 512
    config.test_text = False
    config.gen_attn = "other scenarios"
    config.representation_dim = 512
    config.n_domains = 3

    rep = torch.rand([config.batch_size, config.representation_dim])
    logits = torch.ones([config.batch_size, config.n_classes]) * 0.5

    target_class = torch.randint(low=1, high=config.n_classes - 1, size=[config.batch_size, 1])
    target_dom = torch.randint(low=0, high=config.n_domains, size=[config.batch_size, 1])
    target = torch.cat([target_class, target_dom], dim=-1)

    text = [
        "pick up stick",
        "pick up bat",
        "pick up ball",
        "put down stick",
        "place ball"
    ]

    output_dict = {"logits": logits, "representations": rep, "text": text}

    loss_fn = GenerativeCaption(config)
    loss = loss_fn(output_dict, target)


def test_BODA_CE():
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    class DumbConfig(object):
        pass

    config = DumbConfig()

    config.feat_dim = 2304
    config.representation_dim = 128
    config.n_classes = 7
    config.n_domains = 5
    config.batch_size = 10
    config.labels = ['label_idx', 'scenario_idx', 'source_idx']
    config.boda_omega = 0.1
    config.boda_balance_domain = True
    config.boda_calibrate_distance = True
    config.boda_upsilon = 1.0
    config.boda_dist = "mahalanobis"
    config.stat_update_start_epoch = 0

    config.dist_stats = ["c_d_means"]

    class DumbDataset(object):
        def __init__(self):
            self.domain_counts = torch.randint(low=0, high=100, size=[config.n_classes, config.n_domains])

        def get_class_domain_counts(self):
            return self.domain_counts

    ds = DumbDataset()

    rep = torch.rand([config.batch_size, config.representation_dim])

    logits = torch.ones([config.batch_size, config.n_classes]) * 0.5

    target_class = torch.randint(low=1, high=config.n_classes, size=[config.batch_size, 1])
    target_dom = torch.randint(low=0, high=config.n_domains - 1, size=[config.batch_size, 1])
    target = torch.cat([target_class, target_dom], dim=-1)

    train_rep_mean = torch.rand([config.n_classes, config.n_domains, config.representation_dim])
    train_rep_cov = torch.rand(
        [config.n_classes, config.n_domains, config.representation_dim, config.representation_dim])

    print(logits.shape, rep.shape, train_rep_mean.shape, target.shape)

    output_dict = {"logits": logits, "representations": rep, "c_d_means": train_rep_mean, "c_d_covs": train_rep_cov,
                   "epoch": 10}

    loss_fn = BODA(config, ds)
    loss = loss_fn(output_dict, target)
    print(loss)


def test_CLIP():
    class DumbConfig(object):
        pass

    config = DumbConfig()
    config.n_classes = 7
    config.n_domains = 5
    config.batch_size = 10
    config.labels = ['label_idx', 'scenario_idx', 'source_idx']
    config.representation_dim = (config.batch_size, config.batch_size)
    rep = {}
    config.mix_txt = False
    config.ori = True
    rep["CLIP"] = {}
    rep["CLIP"]["logits_img_video"] = torch.rand([config.batch_size, config.batch_size]).cuda()
    rep["CLIP"]["logits_txt_video"] = torch.rand([config.batch_size, config.batch_size]).cuda()
    rep["CLIP"]["logits_img_text"] = torch.rand([config.batch_size, config.batch_size]).cuda()
    rep["CLIP"]["logits_txt_text"] = torch.rand([config.batch_size, config.batch_size]).cuda()
    rep["CLIP"]["logits_img_ori"] = torch.rand([config.batch_size, config.batch_size]).cuda()
    rep["CLIP"]["logits_txt_ori"] = torch.rand([config.batch_size, config.batch_size]).cuda()

    target_class = torch.randint(low=1, high=config.n_classes - 1, size=[config.batch_size, 1])
    target_dom = torch.randint(low=0, high=config.n_domains, size=[config.batch_size, 1])
    target = torch.cat([target_class, target_dom], dim=-1)
    loss_fn = CLIP(config)
    loss = loss_fn(rep, target)