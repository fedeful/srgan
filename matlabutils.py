from __future__ import print_function
import scipy.io as spio


dataset_path = ''
m = spio.loadmat(dataset_path)

print(m[''][0][4])