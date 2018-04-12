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