from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class VggCutted(nn.Module):
            def __init__(self, vgg_model, number_cutted_layer):
                super(VggCutted, self).__init__()
                self.features = nn.Sequential(
                    *list(vgg_model.features.children())[:number_cutted_layer+1]
                )

            def forward(self, x):
                x = self.features(x)
                return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        tmp = self.prelu(self.bn1(self.conv1(x)))
        return x + self.bn2(self.conv2(tmp))


class SecondGeneratorBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=256, kernel_size=3, stride=1):
        super(SecondGeneratorBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.ps = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.ps(self.conv(x)))


class SRGanGenerator(nn.Module):
    def __init__(self, residual_blocks_number, second_blocks_number):
        super(SRGanGenerator, self).__init__()

        self.rbn = residual_blocks_number
        self.sbn = second_blocks_number

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.pr = nn.PReLU()

        for i in np.arange(0, residual_blocks_number):
            name = 'residual_blocks_number_%03d' % (i)
            self.add_module(name, ResidualBlock())

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=64)

        for i in np.arange(0, second_blocks_number):
            name = 'second_blocks_number_%03d' % (i)
            self.add_module(name, SecondGeneratorBlock())

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):

        x = self.pr(self.conv1(x))

        tmp = x.clone()
        for i in np.arange(0, self.rbn):
            name = 'residual_blocks_number_%03d' % (i)
            tmp = self.__getattr__(name)(tmp)

        x = self.bn(self.conv2(tmp)) + x

        for i in np.arange(0, self.sbn):
            name = 'second_blocks_number_%03d' % (i)
            x = self.__getattr__(name)(x)

        return self.conv3(x)


class SRGanDiscriminator(nn.Module):
    def __init__(self):
        super(SRGanDiscriminator, self).__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.lr = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.lr1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.lr2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.lr3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.lr4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=256)
        self.lr5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=512)
        self.lr6 = nn.LeakyReLU(0.2)

        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=512)
        self.lr7 = nn.LeakyReLU(0.2)

        self.dense1 = nn.Linear(8*8*512, 1024)
        self.lr8 = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(1024, 1)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = self.lr(self.conv(x))

        y = self.lr1(self.conv1(y))
        y = self.lr2(self.conv2(y))
        y = self.lr3(self.conv3(y))
        y = self.lr4(self.conv4(y))
        y = self.lr5(self.conv5(y))
        y = self.lr6(self.conv6(y))
        y = self.lr7(self.conv7(y))

        y = y.view(y.size(0), -1)
        y = self.lr8(self.dense1(y))
        output = self.sig(self.dense2(y))
        return output