import torch
import wandb
from feature_dataset import feature_dataset
import pandas as pd
import umap
import numpy as np

import utils

wandb.init()
config = wandb.config
config.meta_path = "/user/work/qh22492/ego4d/manifest_clean.csv"
config.feat_path = "/user/work/qh22492/ego4d/slowfast_feats_flat"
config.epochs = 20
config.batch_size = 1
config.num_workers = 16
config.frames = 1
config.stride = 1
# config.sample_mode = 'vid_mean'
config.sample_mode = 'rand'
# config.domain = ['video_source', 'device', 'deid_scrubbed_faces', 'scenarios']
config.domain = ['device']
# set to 0 to select videos with any number of scenarios
config.n_scenarios = 1
config.similarity_fns = ['mmd', 'var', 'coral', 'rsd']
config.similarity__balanced = [False, False, False, True]
config.shuffle = True

config.max_samples_to_plot = 99999999
# use to do offline rather than in the wandb webui - much faster and less laggy
config.offline_umap = True
config.umap_nn = 500
config.umap_min_dist = 0.1




ds = feature_dataset(config)
dl = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=config.shuffle)

collected_feats = []
collected_labs = []

if config.sample_mode == 'vid_mean':
    epochs = 1
else:
    epochs = config.epochs

i = 0
for epoch in range(epochs):
    for f, l in dl:
        if i > config.max_samples_to_plot:
            break
        i += 1
        collected_feats.append(f.detach().clone())
        collected_labs.append(l.detach().clone())


collected_feats = torch.stack(collected_feats).squeeze()
collected_labs = torch.stack(collected_labs).squeeze()

# some distribution distance measures may need balanced src and tgt.
class_counts = torch.stack([torch.sum(torch.where(collected_labs == l, 1, 0)) for l in torch.unique(collected_labs)])
min_class_count = torch.min(class_counts)

collected_lab_names = [ds.get_domain_names(l) for l in collected_labs]

# if type(collected_labs[0]) == bool:
#     collected_labs = [str(l) for l in collected_labs]

log_dict = {}
for idx in range(len(config.similarity_fns)):
    sim_fn_name = config.similarity_fns[idx]

    if config.similarity__balanced[idx]:
        class_count = min_class_count
    else:
        class_count = np.iinfo(np.int32).max

    similarity_fn = getattr(utils, sim_fn_name)
    similarities = similarity_fn(collected_feats, collected_labs, max_samples_per_class=class_count)
    doms = ds.get_all_domain_names()[0]
    sim_df = pd.DataFrame(similarities, columns=doms)

    #get mean distance to all other classes, then sort
    mean_sim = np.mean(similarities, axis=0)
    ranking = [doms[i] for i in np.argsort(mean_sim)]
    print("{} ranking: {}".format(sim_fn_name, ranking))

    sim_tab = wandb.Table(dataframe=sim_df)

    sim_hist = wandb.Histogram(similarities.reshape(-1), num_bins=10)
    log_dict[sim_fn_name] = sim_tab
    log_dict[sim_fn_name + "hist"] = similarities

wandb.log(log_dict)


if config.offline_umap:
    embedding = umap.UMAP(n_neighbors=config.umap_nn,
                      min_dist=config.umap_min_dist,
                      metric='correlation').fit_transform(collected_feats)
    collected_feats = torch.tensor(embedding)


cols = [f"dim_{i}" for i in range(collected_feats.shape[-1])]
df = pd.DataFrame(collected_feats, columns=cols)

for i in range(len(collected_lab_names[0])):
    df["LABEL_{}".format(config.domain[i])] = [lab[i] for lab in collected_lab_names]
    
table = wandb.Table(columns=df.columns.to_list(), data=df.values)
wandb.log({"Features": table})



wandb.finish()