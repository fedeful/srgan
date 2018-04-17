from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from models import SRGanDiscriminator, SRGanGenerator, VggCutted
from printutils import print_partial_result, NetworkInfoPrinter, save_partial_result
from torchvision.utils import save_image
import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from splitutils import split_index_train_validation_test
#---------------PARAMETERS---------------#
number_epochs = 2
batch_size = 16
learning_rate = 0.0001
low_res_size = 32
high_res_size = 128
ch_size = 3
res_blocks = 16
up_blocks = 2
cutted_layer_vgg = 5
out_image_flag = True
pre_train_flag = True
continue_adv_train = False
cuda = False
final_path = './weights'
partial_image = './printed_image'
#dataset_folder = '../../remote/datasets/CelebA/'
dataset_folder = 'CelebA/'
random_seed = 46


train_dataset = dsets.ImageFolder(root=dataset_folder,
                                  transform=transforms.ToTensor())

validation_dataset = dsets.ImageFolder(root=dataset_folder,
                                       transform=transforms.ToTensor())

test_dataset = dsets.ImageFolder(root=dataset_folder,
                                 transform=transforms.ToTensor())
validation_size = 0.1
test_size = 0.3

train_sampler, valid_sampler, test_sampler = split_index_train_validation_test(len(train_dataset), 46, test_size, validation_size)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           shuffle=True,
                                           num_workers=2)

valid_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           shuffle=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          sampler=train_sampler,
                                          shuffle=True,
                                          num_workers=2)