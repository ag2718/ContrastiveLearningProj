import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

import numpy as np

from tree import Tree

# FIXME: Define classes hierarchy using class in tree.py, passed in as recursive lis
classification_tree = Tree([])
# in order of one-hot encoding — used for converting one hot encodings of specific classes to contrastive / clf classes
list_of_specific_classes = []

#### IMPORTANT PART #####

# FIXME: Adjust specificity of contrastive classes and clf classes

contrastive_classes_specificity = -1
clf_classes_specificity = -1

#

contrastive_classes = classification_tree.nodes_at_depth(
    contrastive_classes_specificity
)
contrastive_class_to_id = {_cls: i for i, _cls in enumerate(contrastive_classes)}

clf_classes = classification_tree.nodes_at_depth(clf_classes_specificity)
clf_classes_to_id = {_cls: i for i, _cls in enumerate(clf_classes)}


def get_contrastive_one_hots(full_one_hot_matrix):
    class_names = [
        list_of_specific_classes[np.argmax(row)] for row in full_one_hot_matrix
    ]
    contrastive_ids = [
        contrastive_class_to_id[class_name] for class_name in class_names
    ]

    res = np.zeros((full_one_hot_matrix.shape[0], len(contrastive_class_to_id)))
    res[np.arange(full_one_hot_matrix.shape[0]), contrastive_ids] = 1

    return res


def get_clf_one_hots(full_one_hot_matrix):
    class_names = [
        list_of_specific_classes[np.argmax(row)] for row in full_one_hot_matrix
    ]
    clf_ids = [clf_classes_to_id[class_name] for class_name in class_names]

    res = np.zeros((full_one_hot_matrix.shape[0], len(clf_classes_to_id)))
    res[np.arange(full_one_hot_matrix.shape[0]), clf_ids] = 1

    return res
