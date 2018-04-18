from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from torch.optim import Adam
from models import SRGanDiscriminator, SRGanGenerator, VggCutted
from logdatautils.printutils import NetworkInfoPrinter
from torchvision.utils import save_image

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
pre_train_flag = False
continue_adv_train = False
cuda = True
final_path = './weights'
partial_image = './printed_image'
dataset_folder = '../../remote/datasets/CelebA/'


#-------------TRAIN-LOADER---------------#

train_dataset = dsets.ImageFolder(root=dataset_folder,
                                  transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

#---------------TRANSFORM----------------#
'''
STD, MEAN SHOULD BE ADAPT TO THE USED DATASET
'''
print_transform = transforms.Compose([
    transforms.Normalize(mean=[-2.117, -2.035, -1.804],
                         std=[4.366, 4.464, 4.444]),
    transforms.ToPILImage()])


low_res_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(low_res_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

u2 = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])
#-------------NETWORK-SETUP----------------#

g_net = SRGanGenerator(res_blocks, 2)
d_net = SRGanDiscriminator()
#d_net = Discriminator()
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
    vgg16cut.cuda()


#-------GENERATOR-PRE-TRAINING------------#
if pre_train_flag:

    nips.title_line('GENERAOTR PRE-TRAINING')
    for epoch in np.arange(0, number_epochs):
        for i, data in enumerate(train_loader):

            high_resolution_real, _ = data

            low_resolution = torch.FloatTensor(int(high_resolution_real.shape[0]), ch_size, low_res_size, low_res_size)

            for j in np.arange(0, int(high_resolution_real.shape[0])):
                low_resolution[j] = low_res_transform(high_resolution_real[j])
                high_resolution_real[j] = norm(high_resolution_real[j])

            if cuda:
                high_resolution_real = Variable(high_resolution_real).cuda()
                high_resolution_fake = g_net(Variable(low_resolution).cuda())
            else:
                high_resolution_real = Variable(high_resolution_real)
                high_resolution_fake = g_net(Variable(low_resolution))

            g_net.zero_grad()
            content_loss = criterion_1(high_resolution_fake, high_resolution_real)
            content_loss.backward()
            g_optimizer.step()

            if i % 50 == 0:
                tmp_dic = dict()
                tmp_dic['Generator Loss'] = ('%s: %.7f', content_loss)
                nips.log_line(epoch, i, tmp_dic)
                nipf.log_line(epoch, i, tmp_dic)
            '''
              if i % 1000 == 0:
                save_partial_result('pretrain_gen_%d_%d' % (epoch, i),
                                    low_resolution[0].cpu(),
                                    high_resolution_real.data[0].cpu(),
                                    high_resolution_fake.data[0].cpu(),
                                    print_transform)
            '''
            if i % 1000 == 0 and out_image_flag:
                save_image(u2(high_resolution_real.data[0]),
                           'printed_image/high_resolution_real/%s_%s.png' % (epoch, i))
                save_image(u2(high_resolution_fake.data[0]),
                           'printed_image/high_resolution_fake/%s_%s.png' % (epoch, i))
                save_image(u2(low_resolution[0]),
                           'printed_image/low_resolution/%s_%s.png' % (epoch, i))

    nipf.end_print()

    torch.save(g_net.state_dict(), '%s/generator_ptrain.pth' % (final_path))

else:
    if continue_adv_train:

        g_pretrained_weights_dict = torch.load('%s/generator_final.pth' % (final_path))
        d_pretrained_weights_dict = torch.load('%s/discriminator_final.pth' % (final_path))

        g_net.load_state_dict(g_pretrained_weights_dict)
        d_net.load_state_dict(d_pretrained_weights_dict)

    else:
        g_pretrained_weights_dict = torch.load('%s/generator_ptrain.pth' % (final_path))
        g_net.load_state_dict(g_pretrained_weights_dict)



#-----------------ADVERSARIAL-TRAINING---------------#

nipf = NetworkInfoPrinter('./logs/advers_train.txt', number_epochs, len(train_dataset), batch_size)

for epoch in np.arange(0, number_epochs):
    for i, data in enumerate(train_loader):

        high_resolution_real, _ = data

        low_resolution = torch.FloatTensor(int(high_resolution_real.shape[0]), ch_size, low_res_size, low_res_size)
        ones_labels = Variable(torch.ones(int(high_resolution_real.shape[0]), 1))
        zeros_labels = Variable(torch.zeros(int(high_resolution_real.shape[0]), 1))

        for j in np.arange(0, int(high_resolution_real.shape[0])):
            low_resolution[j] = low_res_transform(high_resolution_real[j])
            high_resolution_real[j] = norm(high_resolution_real[j])

        if cuda:
            zeros_labels = zeros_labels.cuda()
            ones_labels = ones_labels.cuda()
            high_resolution_real = Variable(high_resolution_real).cuda()
            high_resolution_fake = g_net(Variable(low_resolution).cuda())
        else:
            high_resolution_real = Variable(high_resolution_real)
            high_resolution_fake = g_net(Variable(low_resolution))

        content_loss = 0
        adversarial_loss = 0

        #----------------DISCRIMINATOR--------------
        d_net.zero_grad()
        d_loss = criterion_2(d_net(high_resolution_real), ones_labels) + criterion_2(d_net(Variable(high_resolution_fake.data)), zeros_labels)
        d_loss.backward()
        d_optimizer.step()


        #-----------------GENERATOR----------------
        g_net.zero_grad()
        tmp = vgg16cut(high_resolution_real)
        real = Variable(vgg16cut(high_resolution_real).data)
        fake = vgg16cut(high_resolution_fake)

        g_content_loss_1 = criterion_1(high_resolution_fake, high_resolution_real)
        g_content_loss_2 = criterion_1(fake, real)
        g_content_loss = g_content_loss_1 + 0.006*g_content_loss_2

        g_adversarial_loss = criterion_2(d_net(high_resolution_fake), ones_labels)

        g_total_loss = g_content_loss + 0.1 * g_adversarial_loss

        g_total_loss.backward()
        g_optimizer.step()

        if i % 100 == 0:
            tmp_dic = dict()
            tmp_dic['Discriminator Loss'] = ('%s: %.7f ', d_loss)
            tmp_dic['Generator Content Loss'] = ('%s: %.7f ', g_content_loss)
            tmp_dic['Generator Adversarial Loss'] = ('%s: %.7f ', g_adversarial_loss)
            tmp_dic['Generator Total Loss'] = ('%s: %.7f ', g_total_loss)
            nips.log_line(epoch, i, tmp_dic)
            nipf.log_line(epoch, i, tmp_dic)

        if i % 1000 == 0 and out_image_flag:
            save_image(u2(high_resolution_real.data[0]),
                       'printed_image/high_resolution_real/adverse_%s_%s.png' % (epoch, i))
            save_image(u2(high_resolution_fake.data[0]),
                       'printed_image/high_resolution_fake/adverse_%s_%s.png' % (epoch, i))
            save_image(u2(low_resolution[0]),
                       'printed_image/low_resolution/adverse_%s_%s.png' % (epoch, i))

nipf.end_print()
nips.end_print()


torch.save(g_net.state_dict(), '%s/generator_final.pth' % (final_path))
torch.save(d_net.state_dict(), '%s/discriminator_final.pth' % (final_path))