import numpy as np
import torch
import wandb
from ffcv.fields import JSONField
import losses
import random


def create_input(use_text, input, device):
    """
        Creates an input dictionary for a given input.

        Args:
            use_text (bool): Flag indicating whether to use text inputs.
            input (tuple): Input tuple containing the necessary data.
            device (torch.device): The device to place the input tensors.

        Returns:
            dict: The input dictionary with the following keys:
                - 'attn': The attention key, set to 'other'.
                - 'data.rgb_feat': The RGB feature data, moved to the specified device.
                - 'data.text': The preprocessed text narration, or None if use_text is False.
                - 'data.target': The target data, moved to the specified device.
        """
    if use_text:
        _, data, narration, noun, target = input
        data = data.to(device)
        narration = preprocess_text(narration)
        target = target.to(device)
    else:
        _, data, _, _, target = input
        data = data.to(device)
        target = target.to(device)
        narration = None
        
    input = {}

    input['data.rgb_feat'] = data
    input['data.text'] = narration
    input['data.target'] = target
    return input


def preprocess_text(text):
    """
        Preprocesses a list of text captions.

        Args:
            text (list): The list of text captions.

        Returns:
            list: The preprocessed list of text captions.
        """
    list_caption = []
    for i in range(len(text)):
        try:
            temp = JSONField.unpack(text[i])
        except:
            temp = '#c is doing something'
        if len(temp) > 0:
            temp = temp[0]
        else:
            print(temp)
        subjects = ['She', 'He', 'The woman', 'The man', 'The user']
        sub = random.choice(subjects)
        t = temp.lower().replace('#unsure', 'something')
        if t[-1] == '.':
            t = t[:-1]
        list_caption.append(t.lower().replace('#c c', sub))
    return list_caption


def init_loss(type, config, dataset):
    return getattr(losses, type)(config, dataset)


def getKey(dct, value):
    return [key for key in dct if (dct[key] == value)][0]


def get_seq_len_from_config(config):
    seq_len = config.n_action_subsample + config.n_before_after_context * 2
    return seq_len


def get_feat_dim_from_config(config):
    feat_dim = config.mlp_hidden_dims[-1]
    return feat_dim


def get_wandb_run_name_from_config_path(config):
    return config.split('/')[-2] + '_' + config.split('/')[-1].split('.')[0]


def calculate_dist_stats(config, representations, labels):
    """
        Calculates distribution statistics for the given representations and labels.

        Args:
            config: The configuration object.
            representations (torch.Tensor): The representations.
            labels (torch.Tensor): The labels.

        Returns:
            dict: A dictionary containing the calculated distribution statistics.
        """
    with torch.no_grad():
        stats_dict = {}

        # only use class (index 0) and one domain (index 1) labels
        labels_scenario = labels[:, 0:2]
        labels_location = labels[:, [0, 2]]
        reps = representations
        dim = representations.shape[-1]

        if "c_d_means" in config.dist_stats:
            c_d_means_scenario = torch.zeros([config.n_classes, config.n_scenarios, dim])
            c_d_means_location = torch.zeros([config.n_classes, config.n_locations, dim])
        if "c_d_vars" in config.dist_stats:
            c_d_vars_scenario = torch.zeros([config.n_classes, config.n_scenarios, dim])
            c_d_vars_location = torch.zeros([config.n_classes, config.n_locations, dim])
        if "c_d_covs" in config.dist_stats:
            c_d_covs_scenario = torch.zeros([config.n_classes, config.n_scenarios, dim, dim])
            c_d_covs_location = torch.zeros([config.n_classes, config.n_locations, dim, dim])

        for c in range(config.n_classes):
            for d in range(config.n_scenarios):
                idxs = torch.where((labels_scenario[:, 0] == c) & (labels_scenario[:, 1] == d), 1, 0)
                idxs = torch.nonzero(idxs)[:, 0]
                if len(idxs) == 0:
                    continue
                c_d_reps = torch.index_select(input=reps, dim=0, index=idxs)
                if "c_d_means" in config.dist_stats:
                    mean = torch.mean(c_d_reps, dim=0)
                    c_d_means_scenario[c, d, :] = mean
                if "c_d_vars" in config.dist_stats:
                    var = torch.var(c_d_reps, dim=0)
                    c_d_vars_scenario[c, d, :] = var
                if "c_d_covs" in config.dist_stats:
                    cov = torch.cov(c_d_reps.t())
                    c_d_covs_scenario[c, d, :, :] = cov
        for c in range(config.n_classes):
            for d in range(config.n_locations):
                idxs = torch.where((labels_location[:, 0] == c) & (labels_location[:, 1] == d), 1, 0)
                idxs = torch.nonzero(idxs)[:, 0]
                if len(idxs) == 0:
                    continue
                c_d_reps = torch.index_select(input=reps, dim=0, index=idxs)
                if "c_d_means" in config.dist_stats:
                    mean = torch.mean(c_d_reps, dim=0)
                    c_d_means_location[c, d, :] = mean
                if "c_d_vars" in config.dist_stats:
                    var = torch.var(c_d_reps, dim=0)
                    c_d_vars_location[c, d, :] = var
                if "c_d_covs" in config.dist_stats:
                    cov = torch.cov(c_d_reps.t())
                    c_d_covs_location[c, d, :, :] = cov
        if "c_d_means" in config.dist_stats:
            stats_dict["c_d_means_scenario"] = c_d_means_scenario
            stats_dict["c_d_means_location"] = c_d_means_location
        if "c_d_vars" in config.dist_stats:
            stats_dict["c_d_vars_scenario"] = c_d_vars_scenario
            stats_dict["c_d_vars_location"] = c_d_vars_location
        if "c_d_covs" in config.dist_stats:
            stats_dict["c_d_covs_scenario"] = c_d_covs_scenario
            stats_dict["c_d_covs_location"] = c_d_covs_location

        return stats_dict


def intersection(lst1, lst2, lst3):
    return list(set(lst1) & set(lst2) & set(lst3))


def accuracy_measures(config, logits, labels):

    _, predictions = torch.max(logits, 1)
    class_labels = labels[:, 0]

    acc = (predictions == class_labels).float().mean()
    class_acc = np.zeros(config.n_classes)
    for c in range(config.n_classes):
        class_acc[c] = ((predictions == class_labels) * (class_labels == c)).float().sum() / max(
            (class_labels == c).sum(), 1)

    mean_class_acc = np.mean([class_acc[int(i)] for i in torch.unique(labels[:, 0])])

    class_names = ["{}".format(i) for i in range(config.n_classes)]
    conf_mat = wandb.plot.confusion_matrix(probs=None, y_true=class_labels.detach().cpu().numpy(),
                                           preds=predictions.detach().cpu().numpy(), class_names=class_names)
    
    wandb.sklearn.plot_confusion_matrix(class_labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(),
                                        labels=None)

    acc_dict = {}
    acc_dict["accuracy"] = acc
    acc_dict["class accuracies"] = class_acc
    acc_dict["mean class accuracy"] = mean_class_acc
    acc_dict["conf mat"] = conf_mat

    return acc_dict

