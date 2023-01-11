import torch
from utils import *

import pytest

def test_mmd():
    dim = 2
    classes = 3
    feat_per_class = 7
    one_feats = torch.ones(feat_per_class, dim)
    one_label = torch.ones(feat_per_class)

    feats = []
    labels = []
    for c in range(classes-1, -1, -1):
        feats.append(c * one_feats)
        labels.append(c * one_label)
    feats = torch.cat(feats)
    labels = torch.cat(labels)

    similarities = mmd(feats, labels)

    assert pytest.approx(similarities[0,2]) == (2**2 + 2**2)**(0.5)


def test_var():
    dim = 2
    classes = 3
    feat_per_class = 7
    one_feats = torch.ones(feat_per_class, dim)
    one_label = torch.ones(feat_per_class)

    feats = []
    labels = []
    for c in range(classes-1, -1, -1):
        feats.append(c * one_feats)
        labels.append(c * one_label)
    feats = torch.cat(feats)
    labels = torch.cat(labels)

    similarities = var(feats, labels)

    assert pytest.approx(similarities[0,2]) == 0

def test_coral():
    dim = 2
    classes = 3
    feat_per_class = 7
    one_feats = torch.ones(feat_per_class, dim)
    one_label = torch.ones(feat_per_class)

    feats = []
    labels = []
    for c in range(classes-1, -1, -1):
        feats.append(c * one_feats)
        labels.append(c * one_label)
    feats = torch.cat(feats)
    labels = torch.cat(labels)

    similarities = coral(feats, labels)

    assert pytest.approx(similarities[0,2]) == 0

def test_rsd():
    dim = 109
    classes = 3
    feat_per_class = 7
    one_feats = torch.ones(feat_per_class, dim)
    one_label = torch.ones(feat_per_class)

    feats_list = []
    labels_list = []
    for c in range(classes-1, -1, -1):
        feats_list.append(c * one_feats)
        labels_list.append(c * one_label)
    feats = torch.cat(feats_list)
    labels = torch.cat(labels_list)

    similarities = rsd(feats, labels)

    # TODO: manually calcuate what this should be
    assert pytest.approx(similarities[0,2]) != 0

    # Now test it works with different number of features per class:
    c = 0
    feats_list.append(c * one_feats)
    labels_list.append(c * one_label)
    feats = torch.cat(feats_list)
    labels = torch.cat(labels_list)
    similarities = rsd(feats, labels)

    # TODO: manually calcuate what this should be
    assert pytest.approx(similarities[0,2]) != 0
    

