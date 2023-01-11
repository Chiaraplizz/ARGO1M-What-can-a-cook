from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, JSONField

from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder
from ffcv.transforms import ToTensor, ToDevice

import wandb
import argparse
import os
import numpy as np
import torch

import sys

sys.path.append("/user/work/qh22492/e4d/")
print(sys.path)
from feature_dataset import classification_dataset
from utils import get_seq_len_from_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", default="/user/home/qh22492/e4d/configs/middle.yaml", help="wandb config file.")
parser.add_argument("--split", type=str, help='Specify split to encode from the config file.')
args = parser.parse_args()

wandb.init(config=args.config, mode="disabled")
config = wandb.config
seq_len = get_seq_len_from_config(config)

split_idx = wandb.config.dataset_splits.index(args.split)
out_fn = config.dataset_ffcvs[split_idx]

ds = classification_dataset(wandb.config, split=args.split, return_numpy=True)
out_path = os.path.join(wandb.config.ffcv_path, out_fn)

feat_d = wandb.config.feat_dim
label_d = len(wandb.config.labels)

print('workers: ', wandb.config.n_workers)
writer = DatasetWriter(out_path, {
    'uid': JSONField(),
    'feat': NDArrayField(shape=(seq_len, feat_d,), dtype=np.dtype('float32')),
    'narration': JSONField(),
    'noun': JSONField(),
    'label': NDArrayField(shape=(label_d,), dtype=np.dtype('int32'))
}, num_workers=wandb.config.n_workers)

writer.from_indexed_dataset(ds, shuffle_indices=wandb.config.ffcv_pre_shuffle)

wandb.finish()