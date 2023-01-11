import torch
import csv
import numpy as np
import pandas as pd
import os
import random


# pd.set_option('display.max_rows', None)


class classification_dataset(torch.utils.data.Dataset):
    """
    use for training an action classifier
    """

    def __init__(self, config, split, return_numpy=False):
        self.config = config
        self.split = split
        self.read_csv()
        self.return_numpy = return_numpy
        print(self.config.feat_path)
        print(self.config.ffcv_path)

    def read_csv(self):
        split_idx = self.config.dataset_splits.index(self.split)
        csv_name = self.config.dataset_csvs[split_idx]
        df = pd.read_csv(os.path.join(self.config.csv_path, csv_name))

        if self.config.csv_len > 0:
            df = df[0:self.config.csv_len]

        self.items = df

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = self.items.iloc[index]
        uid = data['uid']
        narration = (data['narration'],)
        noun = (data['noun'],)
        #text = self.preprocess_text(narration)
        label = [int(data[d]) for d in self.config.labels]
        label_idx = torch.tensor(label)

        feat_path = os.path.join(self.config.feat_path, uid + ".pt")

        with open(feat_path, 'rb') as f:
            feat = torch.load(f)

        start_idx = data['action_start_feature_idx']
        end_idx = data['action_end_feature_idx']
        mid_idx = (data['action_start_feature_idx'] + data['action_end_feature_idx']) / 2
        n_action_subsample = self.config.n_action_subsample
        n_before_after_context = self.config.n_before_after_context
        action_subsample_idxs = np.rint(np.linspace(start_idx, end_idx, num=n_action_subsample)) if (
                n_action_subsample > 1 or self.config.sample_mode == 'start') else [mid_idx]
        context_before = np.array([start_idx - i for i in range(n_before_after_context, 0, -1)])
        context_after = np.array([end_idx + i for i in range(1, n_before_after_context + 1)])
        selection_idxs = np.concatenate([context_before, action_subsample_idxs, context_after])
        selection_idxs = np.clip(selection_idxs, 0, feat.shape[0] - 1)

        feat_selection = feat[selection_idxs]

        if self.return_numpy:
            feat_selection = feat_selection.numpy().astype('float32')
            label_idx = label_idx.numpy().astype('int32')
        return (uid,), feat_selection, narration, noun, label_idx

    def get_n_labels(self):
        return [len(l) for l in self.all_labels]

    def get_class_counts(self):
        class_counts = [len(self.items.loc[self.items['label_idx'] == c]) for c in range(self.config.n_classes)]
        return class_counts

    def get_domain_counts(self):
        raise NotImplementedError()

    def get_class_domain_counts(self):
        counts = torch.zeros([self.config.n_classes, self.config.n_domains])
        for c in range(self.config.n_classes):
            for d in range(self.config.n_domains):
                counts[c, d] = len(
                    self.items.loc[(self.items['label_idx'] == c) & (self.items[self.config.labels[1]] == d)])
        return counts
    def get_class_source_counts(self):
        counts = torch.zeros([self.config.n_classes, self.config.n_sources])
        for c in range(self.config.n_classes):
            for d in range(self.config.n_sources):
                counts[c, d] = len(
                    self.items.loc[(self.items['label_idx'] == c) & (self.items[self.config.labels[1]] == d)])
        return counts

    def get_class_names(self):
        raise NotImplementedError()



class feature_dataset(torch.utils.data.Dataset):
    """
    Use for plotting features
    """

    def __init__(self, config, loc='all', subset='all'):
        self.config = config
        self.loc = loc
        self.subset = subset

        self.read_csv()

    def __getitem__(self, index, frames=1, stride=1, mode='rand'):
        data = self.items.iloc[index]
        uid = data['video_uid']
        domain = [data[d] for d in self.config.domain]
        label = [self.domain2idx[i][domain[i]] for i in range(len(self.config.domain))]
        label = torch.tensor(label)

        feat_path = os.path.join(self.config.feat_path, uid + ".pt")

        with open(feat_path, 'rb') as f:
            feat = torch.load(f)

        if self.config.sample_mode == 'rand':
            high = feat.shape[0] - self.config.frames * self.config.stride - 1
            idx = random.randint(0, high)
            feat_selection = feat[idx: idx + self.config.frames * self.config.stride: self.config.stride]
        elif self.config.sample_mode == 'vid_mean':
            feat_selection = torch.mean(feat, dim=0)

        else:
            raise NotImplementedError()

        return feat_selection, label

    def __len__(self):
        return len(self.items)

    def read_csv(self):

        df = pd.read_csv(self.config.meta_path)

        df['n_scenarios'] = df['scenarios'].str.count('\'') / 2
        if self.config.n_scenarios > 0:
            print("keeping {} out of {} videos with {} scenarios.".format(
                len(df.loc[df['n_scenarios'] == self.config.n_scenarios]), len(df), self.config.n_scenarios))
            df = df.loc[df['n_scenarios'] == self.config.n_scenarios]
        # s_counts = df.loc[df['n_scenarios'] == self.config.n_scenarios]['scenarios'].value_counts()
        # s_counts.plot.bar(figsize=(25, 25)).get_figure().savefig('figs/scenarios.png')

        domains = [df[d].unique() for d in self.config.domain]
        print(domains)

        try:
            domains.sort()
        except:
            pass

        self.all_domains = domains

        self.domain2idx = [{d: i for i, d in enumerate(dom)} for dom in self.all_domains]

        if self.loc == 'all':
            self.items = df
        else:
            raise NotImplementedError()

        if self.subset == "all":
            pass
        else:
            raise NotImplementedError()

    def get_domain_names(self, l):
        if len(self.config.domain) > 1:
            return [self.all_domains[i][l[i]] for i in range(len(l))]
        return [self.all_domains[0][l]]

    def get_all_domain_names(self):
        return self.all_domains


