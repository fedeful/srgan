from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from PIL import Image


class NetworkInfoPrinter:

    def __init__(self, path, epochs, dataset_len, batch_size):
        self.path = path
        self.epochs = epochs
        self.dataset_len = dataset_len
        self.batch_size = batch_size

        if path == '':
            self.fm = False
        else:
            self.fm = True
            self.info_file = open(self.path, 'w')

    def title_line(self, info):
        if self.fm:
            self.info_file.write(info+'\n')
        else:
            print(info)

    def log_line(self, epoch, batch_iteration, dictionary_info):

        info = 'Epoch: [%02d/%02d] ' % (epoch, self.epochs)
        info += 'Iteration: [%02d/%02d] ' % (batch_iteration, self.dataset_len / self.batch_size)
        for k, v in dictionary_info.iteritems():
            info += v[0] % (k, v[1])
        if self.fm:
            self.info_file.write(info+'\n')
        else:
            print(info)

    def end_print(self):

        self.info_file.close()

    def __del__(self):
        if self.fm:
            self.info_file.close()


def save_img(name, path, img, transform):

    result = Image.fromarray((img).astype(np.uint8))
    result.save('%s/%s' % (path, name))


def print_partial_result(low_resolution, high_resolution_real, high_resolution_fake, transform):

    lr_image = np.asarray(transform(low_resolution))
    hrr_image = np.asarray(transform(high_resolution_real))
    hrf_image = np.asarray(transform(high_resolution_fake))

    #figure = plt.figure(figsize=(1, 3), dpi=80)

    #figure.add_subplot(1, 3, 1)
    plt.imshow(lr_image)

    #figure.add_subplot(1, 3, 2)
    plt.imshow(hrr_image)

    #figure.add_subplot(1, 3, 3)
    plt.imshow(hrf_image)

    plt.show()