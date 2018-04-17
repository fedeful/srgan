import os
from skimage import io, transform
import os.path as osp
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms, utils


class RAPDatasetTrain(Dataset):
    """RAP dataset train"""

    def __init__(self, annotations, root_dir, transform=None):
        self.db = RAP(annotations, 0)
        self.labels_name = [db.attr_eng[i][0][0] for i in np.arange(0, db.labels.shape[1])]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.db.train_ind.shape[0])

    def __getitem__(self, idx):
        converted_index = self.db.train_ind[idx]

        relative_image_name = self.db.get_img_path(converted_index)
        absolute_image_name = os.path.join(self.root_dir, relative_image_name)
        image = io.imread(absolute_image_name)

        sample = {'image': image, 'labels': self.db.labels[converted_index]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_labels_name(self):
        return self.labels_name


class RAPDatasetTest(Dataset):
    """RAP dataset test"""

    def __init__(self, annotations, root_dir, transform=None):
        self.db = RAP(annotations, 0)
        self.labels_name = [db.attr_eng[i][0][0] for i in np.arange(0, db.labels.shape[1])]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.db.test_ind.shape[0])

    def __getitem__(self, idx):
        converted_index = self.db.test_ind[idx]

        relative_image_name = self.db.get_img_path(converted_index)
        absolute_image_name = os.path.join(self.root_dir, relative_image_name)
        image = io.imread(absolute_image_name)

        sample = {'image': image, 'labels': self.db.labels[converted_index]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_labels_name(self):
        return self.labels_name


class RAP:
    def __init__(self, db_path, par_set_id):
        self._db_path = db_path

        rap = sio.loadmat(osp.join(self._db_path, 'RAP_annotation', 'RAP_annotation.mat'))['RAP_annotation']

        self._partition = rap[0][0][0]
        self.labels = rap[0][0][1]
        self.attr_ch = rap[0][0][2]
        self.attr_eng = rap[0][0][3]
        self.num_attr = self.attr_eng.shape[0]
        self.position = rap[0][0][4]
        self._img_names = rap[0][0][5]
        self.attr_exp = rap[0][0][6]

        self.attr_group = [range(1, 4), range(4, 7), range(7, 9), range(9, 11), range(30, 36), ]

        self.flip_attr_pairs = [(54, 55)]

        self.expected_loc_centroids = np.ones(self.num_attr, dtype=int) * 2
        self.expected_loc_centroids[9:16] = 1
        self.expected_loc_centroids[35:43] = 1

        self.labels = np.array([[0.5 if x == 2 else x for x in line] for line in self.labels])

        self.test_ind = None
        self.train_ind = None
        self.label_weight = None
        self.set_partition_set_id(par_set_id)

    def set_partition_set_id(self, par_set_id):
        self.train_ind = self._partition[par_set_id][0][0][0][0][0] - 1
        self.test_ind = self._partition[par_set_id][0][0][0][1][0] - 1
        pos_cnt = sum(self.labels[self.train_ind])
        self.label_weight = pos_cnt / self.train_ind.size

    def get_img_path(self, img_id):
        return osp.join(self._db_path, 'RAP_dataset', self._img_names[img_id][0][0])


if __name__ == '__main__':
    db = RAP('../../remote/datasets/', 0)
    print db._partition.shape
    print db._partition[0][0][0][0][1].shape
    print db._partition[1][0][0][0][1].shape
    print "Labels:", db.labels[0]
    print db.test_ind.shape
    print 'Max training index: ', max(db.train_ind)
    print db.get_img_path(0)
    print db.num_attr
    print db.label_weight
    print db.attr_eng[0][0][0]
    print db.attr_eng[1][0][0]
    tdataset = RAPDatasetTrain('../../remote/datasets/', '../../remote/datasets/')
    sample = tdataset[1561]
    plt.imshow(sample['image'])
    plt.show()
    print tdataset.get_labels_name()