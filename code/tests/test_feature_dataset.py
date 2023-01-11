from feature_dataset import feature_dataset, classification_dataset
import wandb

def test_single_domain():

    class DumbConfig(object):
        pass

    config = DumbConfig()
    config.csv_path = "/user/home/qh22492/e4d/csv_files"
    config.feat_path = "/user/work/qh22492/Ego4d/ego4d_data/v1/slowfast8x8_r101_k400"
    config.frames = 1
    config.stride = 1
    config.sample_mode = 'vid_mean'
    config.domain = ['device']

    fl = feature_dataset(config)
    for i in range(3):
        f, s = fl[i]
        name = fl.get_domain_names(s)

def test_mult_domains():

    class DumbConfig(object):
        pass

    config = DumbConfig()
    config.csv_path = "/user/home/qh22492/e4d/csv_files"
    config.feat_path = "/user/work/qh22492/Ego4d/ego4d_data/v1/slowfast8x8_r101_k400"
    config.frames = 1
    config.stride = 1
    config.sample_mode = 'vid_mean'
    config.domain = ['video_source', 'device']

    fl = feature_dataset(config)
    for i in range(3):
        f, s = fl[i]
        name = fl.get_domain_names(s)

def test_rand_seq():

    class DumbConfig(object):
        pass

    config = DumbConfig()
    config.csv_path = "/user/home/qh22492/e4d/csv_files"
    config.feat_path = "/user/work/qh22492/Ego4d/ego4d_data/v1/slowfast8x8_r101_k400"
    config.frames = 1
    config.stride = 1
    config.sample_mode = 'rand'
    config.domain = ['video_source', 'device']

    fl = feature_dataset(config)
    for i in range(3):
        f, s = fl[i]
        name = fl.get_domain_names(s)



def test_scenarios():

    class DumbConfig(object):
        pass

    config = DumbConfig()
    config.csv_path = "/user/home/qh22492/e4d/csv_files"
    config.feat_path = "/user/work/qh22492/Ego4d/ego4d_data/v1/slowfast8x8_r101_k400"
    config.frames = 1
    config.stride = 1
    config.sample_mode = 'vid_mean'
    config.domain = ['video_source', 'scenarios']
    
    # set to 0 to select videos with any number of scenarios
    config.n_scenarios = 1

    fl = feature_dataset(config)
    for i in range(3):
        f, s = fl[i]
        name = fl.get_domain_names(s)


def test_classification_dataset():

    class DumbConfig(object):
        pass

    config = DumbConfig()
    config.csv_path = "/user/work/tp8961/ego4d/data_csvs"
    config.feat_path = "/user/work/tp8961/ego4d/slowfast_feats_flat"
    config.dataset_splits = ["train"]
    config.dataset_csvs = ["train.csv"]
    config.n_classes = 60
    config.frames = 1
    config.stride = 1
    config.sample_mode = 'start'
    config.labels = ['label_idx', 'source_idx', 'scenario_idx']
    config.use_ffcv = False
    config.n_action_subsample = 3
    config.n_before_after_context = 2

    config.csv_len = 128

    fl = classification_dataset(config, split="train")

    assert len(fl) == 128

    data, label = fl[0]
    print(data.shape, label.shape)
    print(data.type(), label.type())

    print(fl.get_class_counts())


