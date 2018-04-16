from __future__ import print_function
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def split_index_train_validation_test(dataset_len, random_seed, test_size, validation_size=0):
    indices = list(range(dataset_len))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    split_validation = 0
    if validation_size != 0:
        split_validation = int(np.floor(validation_size * dataset_len))
    split_train = int(np.floor(test_size * dataset_len))

    if validation_size != 0:
        train_idx, valid_idx, test_idx = indices[split_train:], indices[:split_validation], indices[split_validation:split_train]
        return SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx), SubsetRandomSampler(test_idx)
    else:
        train_idx, test_idx = indices[split_train:], indices[:split_train]
        return SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)
