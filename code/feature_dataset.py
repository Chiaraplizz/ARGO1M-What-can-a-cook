import torch
import numpy as np
import pandas as pd
import os



class classification_dataset(torch.utils.data.Dataset):
    """
    Use for extracting FFCV files.
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

    def get_class_scenario_counts(self):
        counts = torch.zeros([self.config.n_classes, self.config.n_scenarios])
        for c in range(self.config.n_classes):
            for d in range(self.config.n_scenarios):
                counts[c, d] = len(
                    self.items.loc[(self.items['label_idx'] == c) & (self.items[self.config.labels[1]] == d)])
        return counts

    def get_class_location_counts(self):
        counts = torch.zeros([self.config.n_classes, self.config.n_locations])
        for c in range(self.config.n_classes):
            for d in range(self.config.n_locations):
                counts[c, d] = len(
                    self.items.loc[(self.items['label_idx'] == c) & (self.items[self.config.labels[2]] == d)])
        return counts

    def get_class_names(self):
        raise NotImplementedError()


