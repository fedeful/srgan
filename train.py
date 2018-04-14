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
from models import SRGanDiscriminator, SRGanGenerator, VggCutted, Generator
from printutils import print_partial_result, NetworkInfoPrinter, save_partial_result

#---------------PARAMETERS---------------#
number_epochs = 2
batch_size = 16
learning_rate = 0.00001
low_res_size = 32
high_res_size = 128
ch_size = 3
res_blocks = 16
up_blocks = 2
cutted_layer_vgg = 5
cuda = True
final_path = './pesi'
partial_image = './printed_image'
dataset_folder = '../../remote/datasets/CelebA/'



#-------------TRAIN-LOADER---------------#

train_dataset = dsets.ImageFolder(root=dataset_folder,
                                  transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

#---------------TRANSFORM----------------#
'''
STD, MEAN SHOULD BE ADAPT TO THE USED DATASET
'''
print_transform = transforms.Compose([
    transforms.Normalize(mean=[-2.118, -2.036, -1.804],
                         std=[4.367, 4.464, 4.444]),
    transforms.ToPILImage()])


low_res_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(low_res_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])


#-------------NETWORK-SETUP----------------#

g_net = SRGanGenerator(res_blocks, up_blocks)
d_net = SRGanDiscriminator(high_res_size)
print(g_net)
print(d_net)

g_optimizer = Adam(g_net.parameters(), lr=learning_rate)
d_optimizer = Adam(d_net.parameters(), lr=learning_rate)

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16cut = VggCutted(vgg16, 5)

criterion_1 = nn.MSELoss()
criterion_2 = nn.MSELoss()

b_fraction = len(train_dataset)/batch_size
nips = NetworkInfoPrinter('', number_epochs, len(train_dataset), batch_size)
nipf = NetworkInfoPrinter('./logs/g_pretrain.txt', number_epochs, len(train_dataset), batch_size)

if cuda:
    g_net.cuda()
    d_net.cuda()
    criterion_1.cuda()
    criterion_2.cuda()


#-------GENERATOR-PRE-TRAINING------------#
for epoch in np.arange(0, number_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data

        high_resolution_real = images.clone()
        low_resolution = torch.FloatTensor(batch_size, ch_size, low_res_size, low_res_size)

        for j in np.arange(0, batch_size):
            low_resolution[j] = low_res_transform(high_resolution_real[j])
            high_resolution_real[j] = norm(high_resolution_real[j])

        if cuda:
            high_resolution_real = Variable(high_resolution_real).cuda()
            high_resolution_fake = g_net(Variable(low_resolution).cuda())
        else:
            high_resolution_real = Variable(high_resolution_real)
            high_resolution_fake = g_net(Variable(low_resolution))

        criterion_1.zero_grad()
        content_loss = criterion_1(high_resolution_fake, high_resolution_real)
        content_loss.backward()
        g_optimizer.step()

        if i % 50 == 0:
            tmp_dic = dict()
            tmp_dic['Generator Loss'] = ('%s: %.7f', content_loss)
            nips.log_line(epoch, i, tmp_dic)
            nipf.log_line(epoch, i, tmp_dic)

        if i % 1000 == 0:
            save_partial_result('pretrain_gen_%d_%d' % (epoch, i),
                                low_resolution[0].cpu(),
                                high_resolution_real.data[0].cpu(),
                                high_resolution_fake.data[0].cpu(),
                                print_transform)

nipf.end_print()
nips.end_print()


#-----------------ADVERSARIAL-TRAINING---------------#
nips = NetworkInfoPrinter('', number_epochs, len(train_dataset), batch_size)
nipf = NetworkInfoPrinter('./logs/advers_train.txt', number_epochs, len(train_dataset), batch_size)

ones_labels = Variable(torch.ones(batch_size, 1))
zeros_labels = Variable(torch.ones(batch_size, 1))

if cuda:
    ones_labels = ones_labels.cuda()
    zeros_labels = zeros_labels.cuda()


for epoch in np.arange(0, number_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data

        high_resolution_real = images.clone()
        low_resolution = torch.FloatTensor(batch_size, ch_size, low_res_size, low_res_size)

        for j in np.arange(0, batch_size):
            low_resolution[j] = low_res_transform(high_resolution_real[j])
            high_resolution_real[j] = norm(high_resolution_real[j])

        if cuda:
            high_resolution_real = Variable(high_resolution_real).cuda()
            high_resolution_fake = g_net(Variable(low_resolution).cuda())
        else:
            high_resolution_real = Variable(high_resolution_real)
            high_resolution_fake = g_net(Variable(low_resolution))

        content_loss = 0
        adversarial_loss = 0

        #----------------DISCRIMINATOR--------------
        d_net.zero_grad()
        d_loss = criterion_2(d_net(high_resolution_real), ones_labels) +\
                             criterion_2(d_net(Variable(high_resolution_fake.data)), zeros_labels)
        d_loss.backward()
        d_optimizer.step()


        #-----------------GENERATOR----------------
        g_net.zero_grad()
        tmp = vgg16cut(high_resolution_real)
        real = Variable(vgg16cut(high_resolution_real).data)
        fake = vgg16cut(high_resolution_fake)

        g_content_loss = criterion_1(high_resolution_fake, high_resolution_real) + 0.006*criterion_1(fake, real)

        g_adversarial_loss = criterion_2(d_net(high_resolution_fake), ones_labels)

        g_total_loss = g_content_loss + 0.001 * g_adversarial_loss

        g_total_loss.backward()
        g_optimizer.step()

        if i % 100 == 0:
            tmp_dic = dict()
            tmp_dic['Discriminator Loss'] = ('%s: %.7f', d_loss)
            tmp_dic['Generator Content Loss'] = ('%s: %.7f', g_content_loss)
            tmp_dic['Generator Adversarial Loss'] = ('%s: %.7f', g_adversarial_loss)
            tmp_dic['Generator Total Loss'] = ('%s: %.7f', g_total_loss)
            nips.log_line(epoch, i, tmp_dic)
            nipf.log_line(epoch, i, tmp_dic)

nipf.end_print()
nips.end_print()


torch.save(g_net.state_dict(), '%s/generator_final.pth' % (final_path))
torch.save(d_net.state_dict(), '%s/discriminator_final.pth' % (final_path))



