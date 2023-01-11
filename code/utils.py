import numpy as np
import torch
import wandb
from ffcv.fields import JSONField
import random
import losses
import torch.nn as nn
from dataclasses import replace
import random
import time

templates_verb = [
    'a bad video of a person that {}s.',
    'a good video of a person that {}s.',
    'a bright video of a person that {}s.',
    'a dark video of a person that {}s.',
    'a blurry video of a person that {}s.',
    'She {}s.',
    'He {}s.',
    'The woman {}s.',
    'The man {}s.',
    'The user {}s.',
    'The subject {}s.',
    'The person {}s.'
]

templates_noun = [
    'a bad video containing {}.',
    'a good video containing {}.',
    'a bright video containing {}.',
    'a dark video containing {}.',
    'a blurry video containing {}.',
    'the video contains {}.',

]
templates_verb_source = [
    'a bad video of a person that {}s in {}.',
    'a good video of a person that {}s in {}.',
    'a bright video of a person that {}s in {}.',
    'a dark video of a person that {}s in {}.',
    'a blurry video of a person that {}s in {}.',
    'She {}s in {}.',
    'He {}s in {}.',
    'The woman {}s in {}.',
    'The man {}s in {}.',
    'The user {}s in {}.',
    'The subject {}s in {}.',
    'The person {}s in {}.'
]

templates_verb_scenario = [
    'a bad video of a person that {}s in {} scenario.',
    'a good video of a person that {}s in {} scenario.',
    'a bright video of a person that {}s in {} scenario.',
    'a dark video of a person that {}s in {} scenario.',
    'a blurry video of a person that {}s in {} scenario.',
    'She {}s in {} scenario.',
    'He {}s in {} scenario.',
    'The woman {}s in {} scenario.',
    'The man {}s in {} scenario.',
    'The user {}s in {} scenario.',
    'The subject {}s in {} scenario.',
    'The person {}s in {} scenario.'
]

templates_verb_source_scenario = [
    'a bad video of a person that {}s in {} in {} scenario.',
    'a good video of a person that {}s in {} in {} scenario.',
    'a bright video of a person that {}s in {} in {} scenario.',
    'a dark video of a person that {}s in {} in {} scenario.',
    'a blurry video of a person that {}s in {} in {} scenario.',
    'She {}s in {} in {} scenario.',
    'He {}s in {} in {} scenario.',
    'The woman {}s in {} in {} scenario.',
    'The man {}s in {} in {} scenario.',
    'The user {}s in {} in {} scenario.',
    'The subject {}s in {} in {} scenario.',
    'The person {}s in {} in {} scenario.'
]

templates_caption_source_scenario = [
    'In a video in {} and recording someone {}, {}.',
    'In a good video of source {} and showing someone {}, {}.',
    'In a bright video of source {} and recording someone {}, {}.',
    'In a dark video of source {} and showing someone {}, {}.',
    'In a video set in {} and recording someone {}, {}.',
    'In a video located in {} and showing someone {}, {}.',
    'In a video with people from {} and {}, {}.',
]

templates_caption_scenario = [
    'In a video showing someone {}, {}.',
    'In a good video showing someone {}, {}.',
    'In a bright video showing someone {}, {}.',
    'In a dark video set while {}, {}.',
    'In a video set while {}, {}.',
    'In a video done while {}, {}.',
    'In a video with people {} in it, {}.',
    'While {}, {}.',
]
'''
templates_caption_scenario = [
    '{} in a video showing someone {}.',
    '{} in a good video showing someone {}.',
    '{} in a bright video showing someone {}.',
    '{} in a dark video set while {}.',
    '{} in a video set while {}.',
    '{} in a video done while {}.',
    '{} in a video with people {} in it.',
    '{} while {}.',
]
'''

templates_caption_source = [
    'In a video in {}, {}.',
    'In a good video in {}, {}.',
    'In a bright video in {}, {}.',
    'In a dark video in {}, {}.',
    'In a video set in {}, {}.',
    'In a video located in {}, {}.',
    'In a video with people in {}, {}.',
    'While being in {}, {}.',
]

templates_verb_noun = [
    'a bad video of a person that {}s the {}.',
    'a good video of a person that {}s the {}.',
    'a bright video of a person that {}s the {}.',
    'a dark video of a person that {}s the {}.',
    'a blurry video of a person that {}s the {}.',
    'She {}s the {}.',
    'He {}s the {}.',
    'The woman {}s the {}.',
    'The man {}s the {}.',
    'The user {}s the {}.',
    'The subject {}s the {}.',
    'The person {}s the {}.'
]

templates_verb_noun_scenario = [
    'a bad video of a person that {}s the {} in {} scenario.',
    'a good video of a person that {}s the {} in {} scenario.',
    'a bright video of a person that {}s the {} in {} scenario.',
    'a dark video of a person that {}s the {} in {} scenario.',
    'a blurry video of a person that {}s the {} in {} scenario.',
    'She {}s the {} in {} scenario.',
    'He {}s the {} in {} scenario.',
    'The woman {}s the {} in {} scenario.',
    'The man {}s the {} in {} scenario.',
    'The user {}s the {} in {} scenario.',
    'The subject {}s the {} in {} scenario.',
    'The person {}s the {} in {} scenario.'
]

templates_verb_noun_source = [
    'a bad video of a person that {}s the {} in {}.',
    'a good video of a person that {}s the {} in {}.',
    'a bright video of a person that {}s the {} in {}.',
    'a dark video of a person that {}s the {} in {}.',
    'a blurry video of a person that {}s the {} in {}.',
    'She {}s the {} in {}.',
    'He {}s the {} in {}.',
    'The woman {}s the {} in {}.',
    'The man {}s the {} in {}.',
    'The user {}s the {} in {}.',
    'The subject {}s the {} in {}.',
    'The person {}s the {} in {}.'
]

templates_verb_noun_source_scenario = [
    'a bad video of a person that {}s the {} in {} in {} scenario.',
    'a good video of a person that {}s the {} in {} in {} scenario.',
    'a bright video of a person that {}s the {} in {} in {} scenario.',
    'a dark video of a person that {}s the {} in {} in {} scenario.',
    'a blurry video of a person that {}s the {} in {} in {} scenario.',
    'She {}s the {} in {} in {} scenario.',
    'He {}s the {} in {} in {} scenario.',
    'The woman {}s the {} in {} in {} scenario.',
    'The man {}s the {} in {} in {} scenario.',
    'The user {}s the {} in {} in {} scenario.',
    'The subject {}s the {} in {} in {} scenario.',
    'The person {}s the {} in {} in {} scenario.'
]

source_dict = {'Pittsburgh': 0,
               'India': 1,
               'Bristol': 2,
               'Singapore': 3,
               'Catania': 4,
               'Georgia': 5,
               'Africa': 6,
               'Saudi Arabia': 7,
               'Facebook': 8,
               'Indiana': 9,
               'Tokyo': 10,
               'Minnesota': 11,
               'Colombia': 12}

scenario_dict = {'Doing sport': 0,
                 'Crafting': 1,
                 'Playing': 2,
                 'Cooking': 3,
                 'Gardening': 4,
                 'Doing a mechanical work': 5,
                 'Cleaning': 6,
                 'Building': 7,
                 'Shopping': 8,
                 'Doing multiple things': 9
                 }

verb_dict = {'dip': 0,
             'hold': 1,
             'paint': 2,
             'read': 3,
             'lift': 4,
             'touch': 5,
             'pat': 6,
             'take': 7,
             'put': 8,
             'wash': 9,
             'open': 10,
             'close': 11,
             'turn-on': 12,
             'cut': 13,
             'pour': 14,
             'mix': 15,
             'remove': 16,
             'throw': 17,
             'shake': 18,
             'scoop': 19,
             'squeeze': 20,
             'peel': 21,
             'press': 22,
             'flip': 23,
             'turn-off': 24,
             'turn': 25,
             'scrape': 26,
             'fold': 27,
             'scrubs': 28,
             'break': 29,
             'pull': 30,
             'eat': 31,
             'wrap': 32,
             'unroll': 33,
             'hang': 34,
             'sprinkle': 35,
             'rip': 36,
             'stretch': 37,
             'divide': 38,
             'rub': 39,
             'brush': 40,
             'drop': 41,
             'drink': 42,
             'gather': 43,
             'attach': 44,
             'wear': 45,
             'lower': 46,
             'screw': 47,
             'carry': 48,
             'bring': 49,
             'bend': 50,
             'spread': 51,
             'give': 52,
             'drag': 53,
             'draw': 54,
             'drill': 55,
             'loosen': 56,
             'unwrap': 57,
             'roll': 58,
             'clear': 59}


def create_input(config, use_text, input, device):
    if use_text:
        uid, data, narration, noun, target = input
        data = data.to(device)
        # uid = preprocess_text(uid)

        narration = preprocess_text(narration)
        if config.use_prompt:
            narration = build_prompt(target[:, 0], noun, target[:, 1], target[:, 2], config)
        # narration = narration.to(device)
        # noun = noun.to(device)
        if config.random_guess:
            target = torch.randint(0, 60, target.shape).to(device)
        target = target.to(device)
        #print(narration)
    else:
        uid, data, _, _, target = input
        # uid = preprocess_text(uid)
        if config.random_guess:
            target = torch.randint(0, 60, target.shape).to(device)
        data = data.to(device)
        target = target.to(device)
        # data.copy_(data, non_blocking=True)
        # target.copy_(target, non_blocking=True)

        narration = None
        noun = None

    target = torch.cat((target, uid.to(device
                                       )), dim=1)
    input = {}

    input['attn'] = 'other'
    input['data.uid'] = uid#[JSONField.unpack(uid)[0] for i in range(len(uid))]
    input['data.rgb_feat'] = data
    input['data.text'] = narration
    input['data.target'] = target
    # input['data.noun'] = build_prompt_noun(noun)

    return input


def build_prompt_noun(noun):
    text_list = []
    for i in range(len(noun)):
        text = ''
        if len(noun[i]) > 0:
            try:
                noun_txt = extract_noun(JSONField.unpack(noun[i]))
            except:
                noun_txt = []
            temp = templates_noun
            i = random.randint(0, len(temp) - 1)
            template = temp[i]
            for j, n in enumerate(noun_txt):
                if noun_txt[j] == 'c':
                    noun_txt[j] = 'a person'
                if noun_txt[j] == 'c':
                    noun_txt[j] = 'something'
                if len(noun_txt) == 1:
                    text = text + n
                elif j == (len(noun_txt) - 1):
                    text = text + 'and ' + n
                else:
                    text = text + n + ', '
            text_list.append(template.format(text))
    return text_list


def preprocess_text(text):
    list_caption = []
    for i in range(len(text)):
        try:
            temp = JSONField.unpack(text[i])
            # temp = text[i].tobytes().decode('ascii').strip()
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
    # print(list_caption)
    return list_caption


def init_loss(type, config, dataset):
    return getattr(losses, type)(config, dataset)


def build_prompt_from_caption(caption, scenario, source, config):
    list_caption = []
    for i in range(len(caption)):
        scenario_txt = extract_scenario(scenario[i]).lower()
        source_txt = extract_source(source[i])
        if 'source' in config.prompt and 'scenario' in config.prompt:
            temp = templates_caption_source_scenario
            j = random.randint(0, len(temp) - 1)
            template = temp[j]
            prompt = template.format(source_txt, scenario_txt, caption[i].lower())
        elif 'source' in config.prompt:
            temp = templates_caption_source
            j = random.randint(0, len(temp) - 1)
            template = temp[j]
            prompt = template.format(source_txt, caption[i].lower())
        elif 'scenario' in config.prompt:
            temp = templates_caption_scenario
            j = random.randint(0, len(temp) - 1)
            template = temp[j]
            prompt = template.format(scenario_txt, caption[i].lower())

        list_caption.append(prompt)

    return list_caption


def build_prompt(verb, noun, scenario, source, config):
    list_caption = []
    for i in range(len(verb)):
        verb_txt = extract_verb(verb[i]).lower()
        if len(noun[i]) > 0:
            try:
                noun_txt = extract_noun(JSONField.unpack(noun[i]))
            except:
                noun_txt = []
        else:
            noun_txt = []
        scenario_txt = extract_scenario(scenario[i]).lower()
        source_txt = extract_source(source[i]).lower()
        if len(noun_txt) == 0 or 'noun' not in config.prompt:
            if 'source' in config.prompt and 'scenario' in config.prompt:
                temp = templates_verb_source_scenario
                i = random.randint(0, len(temp) - 1)
                template = temp[i]
                prompt = template.format(verb_txt, source_txt, scenario_txt)
            elif 'source' in config.prompt:
                temp = templates_verb_source
                i = random.randint(0, len(temp) - 1)
                template = temp[i]
                prompt = template.format(verb_txt, source_txt)
            elif 'scenario' in config.prompt:
                temp = templates_verb_scenario
                i = random.randint(0, len(temp) - 1)
                template = temp[i]
                prompt = template.format(verb_txt, scenario_txt)
            else:
                temp = templates_verb
                i = random.randint(0, len(temp) - 1)
                template = temp[i]
                prompt = template.format(verb_txt)
            list_caption.append(prompt)
        else:
            if 'source' in config.prompt and 'scenario' in config.prompt:
                temp = templates_verb_noun_source_scenario
                i = random.randint(0, len(temp) - 1)
                template = temp[i]
                prompt = template.format(verb_txt, noun_txt[0], source_txt, scenario_txt)
            elif 'source' in config.prompt:
                temp = templates_verb_noun_source
                i = random.randint(0, len(temp) - 1)
                template = temp[i]
                prompt = template.format(verb_txt, noun_txt[0], source_txt)
            elif 'scenario' in config.prompt:
                temp = templates_verb_noun_scenario
                i = random.randint(0, len(temp) - 1)
                template = temp[i]
                prompt = template.format(verb_txt, noun_txt[0], scenario_txt)
            else:
                temp = templates_verb_noun
                i = random.randint(0, len(temp) - 1)
                template = temp[i]
                prompt = template.format(verb_txt, noun_txt[0])
            list_caption.append(prompt)
    return list_caption


def extract_verb(verb):
    return getKey(verb_dict, verb)


def extract_noun(noun):
    list = []
    import ast
    noun_list = ast.literal_eval(noun[0])
    for n in noun_list:
        if ':' not in n:
            list.append(n)
        else:
            list.append(n.split(':')[1] + ' ' + n.split(':')[0])
    return list


def extract_source(source):
    return getKey(source_dict, source)


def extract_scenario(scenario):
    return getKey(scenario_dict, scenario)


def getKey(dct, value):
    return [key for key in dct if (dct[key] == value)][0]


def get_seq_len_from_config(config):
    seq_len = config.n_action_subsample + config.n_before_after_context * 2
    return seq_len


def get_feat_dim_from_config(config):
    if config.model == 'MLP':
        feat_dim = config.mlp_hidden_dims[-1]
    elif config.model == 'TRN':
        feat_dim = config.trn_bottleneck * config.trn_scales
    elif config.model == 'Transformer':
        feat_dim = config.tfmr_d_model
    return feat_dim


def get_wandb_run_name_from_config_path(config):
    return config.split('/')[-2] + '_' + config.split('/')[-1].split('.')[0]


def calculate_dist_stats(config, representations, labels):
    with torch.no_grad():
        stats_dict = {}

        # only use class (index 0) and one domain (index 1) labels
        labels_activity = labels[:, 0:2]
        labels_source = labels[:, [0, 3]]
        reps = representations
        dim = representations.shape[-1]

        if "c_d_means" in config.dist_stats:
            c_d_means = torch.zeros([config.n_classes, config.n_domains, dim])
            c_d_means_source = torch.zeros([config.n_classes, config.n_sources, dim])
        if "c_d_vars" in config.dist_stats:
            c_d_vars = torch.zeros([config.n_classes, config.n_domains, dim])
        if "c_d_covs" in config.dist_stats:
            c_d_covs = torch.zeros([config.n_classes, config.n_domains, dim, dim])

        for c in range(config.n_classes):
            for d in range(config.n_domains):
                idxs = torch.where((labels_activity[:, 0] == c) & (labels_activity[:, 1] == d), 1, 0)
                idxs = torch.nonzero(idxs)[:, 0]
                if len(idxs) == 0:
                    continue
                c_d_reps = torch.index_select(input=reps, dim=0, index=idxs)
                if "c_d_means" in config.dist_stats:
                    mean = torch.mean(c_d_reps, dim=0)
                    c_d_means[c, d, :] = mean
                if "c_d_vars" in config.dist_stats:
                    var = torch.var(c_d_reps, dim=0)
                    c_d_vars[c, d, :] = var
                if "c_d_covs" in config.dist_stats:
                    cov = torch.cov(c_d_reps.t())
                    c_d_covs[c, d, :, :] = cov
        for c in range(config.n_classes):
            for d in range(config.n_sources):
                idxs = torch.where((labels_source[:, 0] == c) & (labels_source[:, 1] == d), 1, 0)
                idxs = torch.nonzero(idxs)[:, 0]
                if len(idxs) == 0:
                    continue
                c_d_reps = torch.index_select(input=reps, dim=0, index=idxs)
                if "c_d_means" in config.dist_stats:
                    mean = torch.mean(c_d_reps, dim=0)
                    c_d_means_source[c, d, :] = mean
                if "c_d_vars" in config.dist_stats:
                    var = torch.var(c_d_reps, dim=0)
                    c_d_vars[c, d, :] = var
                if "c_d_covs" in config.dist_stats:
                    cov = torch.cov(c_d_reps.t())
                    c_d_covs[c, d, :, :] = cov
        if "c_d_means" in config.dist_stats:
            stats_dict["c_d_means"] = c_d_means
            stats_dict["c_d_means_source"] = c_d_means_source
        if "c_d_vars" in config.dist_stats:
            stats_dict["c_d_vars"] = c_d_vars
        if "c_d_covs" in config.dist_stats:
            stats_dict["c_d_covs"] = c_d_covs

        return stats_dict


def intersection(lst1, lst2, lst3):
    return list(set(lst1) & set(lst2)& set(lst3))
def accuracy_measures(config, logits, labels, split):
    correct_pred = np.zeros(config.n_classes)
    total_pred = np.zeros(config.n_classes)

    _, predictions = torch.max(logits, 1)
    class_labels = labels[:, 0]

    acc = (predictions == class_labels).float().mean()
    class_acc = np.zeros(config.n_classes)
    for c in range(config.n_classes):
        class_acc[c] = ((predictions == class_labels) * (class_labels == c)).float().sum() / max(
            (class_labels == c).sum(), 1)
    print('Acc put: ', class_acc[8])
    print('Acc drop: ', class_acc[42])
    #return percentage class_label is 8 and predictions is 42
    print('put(drop): ', ((class_labels == 8) * (predictions == 42)).sum() / max(
        (class_labels == 8).sum(), 1))
    print('drop(put): ', ((class_labels == 42) * (predictions == 8)).sum() / max(
        (class_labels == 42).sum(), 1))
    
    mean_class_acc = np.mean([class_acc[int(i)] for i in torch.unique(labels[:, 0])])
    # mean_class_acc = np.mean(class_acc)

    class_names = ["{}".format(i) for i in range(config.n_classes)]
    conf_mat = wandb.plot.confusion_matrix(probs=None, y_true=class_labels.detach().cpu().numpy(),
                                           preds=predictions.detach().cpu().numpy(), class_names=class_names)
    if config.resume and False:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        action_label_dict = {v: k for k, v in verb_dict.items()}

        plt.figure(figsize=(30, 30))

        ax = plt.subplot()
        cm = confusion_matrix(class_labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(),
                              labels=[i for i in range(60)])
        selected_classes = sorted([7, 8, 41, 1, 16, 13, 5, 10, 2, 9, 4, 14, 17, 59, 19])

        cm_selected = cm[[i for i in selected_classes], :]
        cm_selected = cm_selected[:, [i for i in selected_classes]]
        classes = [i for i in range(config.n_classes)]
        cm_other = cm[[x for x in classes if x not in selected_classes], :]
        cm_other = cm_other[:, [i for i in selected_classes]]
        row = cm_other.sum(axis=0)
        cm_other = cm[:, [x for x in classes if x not in selected_classes]]
        cm_other = cm_other[[i for i in selected_classes], :]

        col = [sum(x) for x in cm_other]
        cm_other = cm[:, [x for x in classes if x not in selected_classes]]
        cm_other = cm_other[[x for x in classes if x not in selected_classes], :]
        element = np.trace(cm_other)
        cm_selected_1 = np.concatenate((cm_selected, np.array([row])), axis=0)
        cm_selected = np.column_stack((cm_selected_1, np.array(col + [element])))

        sns.heatmap(cm_selected, annot=True, fmt='g', ax=ax,
                    xticklabels=[action_label_dict[i] for i in selected_classes] + [0],  # [i.item() for i in
                    # torch.sort(torch.unique(torch.cat((predictions, labels[:, 0]), dim=0)))[0]],
                    yticklabels=[action_label_dict[i] for i in selected_classes] + [
                        0])  # [i.item() for i in torch.sort(torch.unique(torch.cat((predictions, labels[:, 0]),
        #                       dim=0)))[
        # 0]])  # annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks

        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        # ax.xaxis.set_ticklabels(class_names)
        # ax.yaxis.set_ticklabels(class_names)
        ax.set_title('Confusion Matrix')

        plt.savefig("Confusion_Matrix_" + split + ".png")
    # print(class_labels)
    # print(predictions)
    wandb.sklearn.plot_confusion_matrix(class_labels.detach().cpu().numpy(), predictions.detach().cpu().numpy(),
                                        labels=None)

    acc_dict = {}
    acc_dict["accuracy"] = acc
    acc_dict["class accuracies"] = class_acc
    acc_dict["mean class accuracy"] = mean_class_acc
    acc_dict["conf mat"] = conf_mat

    return acc_dict


def mmd(feats, labels, max_samples_per_class=np.iinfo(np.int32).max):
    feats = feats.numpy()
    labels = labels.numpy()
    labels = labels.astype(int)
    unique_labels = np.unique(labels)

    max_lab = np.max(labels) + 1

    means = [0] * max_lab
    for l in unique_labels:
        l_idxs = np.where(labels == l)
        l_feats = feats[l_idxs]
        if l_feats.shape[0] > max_samples_per_class:
            l_feats = l_feats[:max_samples_per_class]

        l_mean = np.mean(l_feats, axis=0)
        means[l] = l_mean

    distances = np.zeros([max_lab, max_lab])

    for i in range(max_lab):
        for j in range(max_lab):
            distances[i, j] = np.linalg.norm(means[i] - means[j])

    return distances


def var(feats, labels, max_samples_per_class=np.iinfo(np.int32).max):
    feats = feats.numpy()
    labels = labels.numpy()
    labels = labels.astype(int)
    unique_labels = np.unique(labels)

    max_lab = np.max(labels) + 1

    covs = [0] * max_lab
    for l in unique_labels:
        l_idxs = np.where(labels == l)
        l_feats = feats[l_idxs]
        if l_feats.shape[0] > max_samples_per_class:
            l_feats = l_feats[:max_samples_per_class]

        l_cov = np.var(l_feats, axis=0)
        covs[l] = l_cov

    distances = np.zeros([max_lab, max_lab])

    for i in range(max_lab):
        for j in range(max_lab):
            distances[i, j] = np.linalg.norm(covs[i] - covs[j])

    return distances


def coral(feats, labels, max_samples_per_class=np.iinfo(np.int32).max):
    feats = feats.numpy()
    labels = labels.numpy()
    labels = labels.astype(int)
    unique_labels = np.unique(labels)

    max_lab = np.max(labels) + 1

    covs = [0] * max_lab
    for l in unique_labels:
        l_idxs = np.where(labels == l)
        l_feats = feats[l_idxs]
        if l_feats.shape[0] > max_samples_per_class:
            l_feats = l_feats[:max_samples_per_class]

        l_cov = np.cov(l_feats, rowvar=False)
        covs[l] = l_cov

    distances = np.zeros([max_lab, max_lab])

    for i in range(max_lab):
        for j in range(max_lab):
            distances[i, j] = np.linalg.norm(covs[i] - covs[j])

    return distances


def rsd(feats, labels, max_samples_per_class=np.iinfo(np.int32).max):
    # set as in paper
    tradeoff2 = 0.01

    labels = labels.numpy()
    labels = labels.astype(int)
    unique_labels = np.unique(labels)

    max_lab = np.max(labels) + 1

    all_u = [0] * max_lab
    all_s = [0] * max_lab
    all_v = [0] * max_lab
    for l in unique_labels:
        l_idxs = np.where(labels == l)[0]
        l_idxs = torch.tensor(l_idxs, dtype=torch.int)

        l_feats = torch.index_select(feats, 0, l_idxs)
        if l_feats.shape[0] > max_samples_per_class:
            fl_feats = l_feats[:max_samples_per_class]

        u, s, v = torch.svd(l_feats.t(), some=True)

        all_u[l] = u
        all_s[l] = s
        all_v[l] = v

    distances = np.zeros([max_lab, max_lab])

    for i in range(max_lab):
        for j in range(max_lab):

            # added to avoid sqrt of a negative number
            epsilon = 0.00001

            # i == j will give NAN
            if i == j:
                continue

            u_s = all_u[i]
            s_s = all_s[i]
            v_s = all_v[i]

            u_t = all_u[j]
            s_t = all_s[j]
            v_t = all_v[j]

            p_s, cospa, p_t = torch.svd(torch.mm(u_s.t(), u_t))

            sinpa = torch.sqrt(1 - torch.pow(cospa, 2) + epsilon)

            # #drop dimensions of largest sample:
            n_dim = min(p_s.shape[0], p_t.shape[0])
            p_s = p_s[:n_dim]
            p_t = p_t[:n_dim]

            distances[i, j] = torch.norm(sinpa, 1) + tradeoff2 * torch.norm(torch.abs(p_s) - torch.abs(p_t), 2)

    return distances

