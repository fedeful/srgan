from __future__ import print_function
import scipy.io as spio


dataset_path = '/home/federico/remote/datasets/RAP_annotation/RAP_annotation.mat'
m = spio.loadmat(dataset_path)

print(m['RAP_annotation'][0][4])