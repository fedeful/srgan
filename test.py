from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from models import SRGanDiscriminator, SRGanGenerator, VggCutted

number_epochs = 1
batch_size = 1
learning_rate = 0.0001
low_res_size = 32
high_res_size = 128
ch_size = 3
cutted_layer_vgg = 5
cuda = False
final_path = './pesi'
partial_image = './printed_image'


g_pretrained_weights_dict = torch.load('%s/generator_final.pth' % (final_path))
d_pretrained_weights_dict = torch.load('%s/discriminator_final.pth' % (final_path))

g_net = SRGanGenerator(5, 2)
d_net = SRGanDiscriminator(high_res_size)

g_net.load_state_dict(g_pretrained_weights_dict)
d_net.load_state_dict(d_pretrained_weights_dict)

g_net.eval()
d_net.eval()


test_dataset = dsets.ImageFolder(root='./CelebA/',
                                  transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset= test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

print_transform = transforms.Compose([transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
                                                           transforms.ToPILImage()])

scale_low_res = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(low_res_size),
                                   transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                   ])

norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16cut = VggCutted(vgg16, 5)

ones_labels = Variable(torch.ones(batch_size, 1))
zeros_labels = Variable(torch.ones(batch_size, 1))

totloss =0.0

criterion_1 = nn.MSELoss()
criterion_2 = nn.MSELoss()

if cuda:
    g_net.cuda()
    d_net.cuda()
    criterion_1.cuda()
    criterion_2.cuda()

for i, data in enumerate(test_loader):

    images, labels = data

    high_resolution_real = images.clone()
    low_resolution_real = torch.FloatTensor(batch_size, 3, low_res_size, low_res_size)

    for j in np.arange(0, batch_size):
        low_resolution_real[j] = scale_low_res(high_resolution_real[j])
        high_resolution_real[j] = norm(high_resolution_real[j])

    if cuda:
        high_resolution_real = Variable(high_resolution_real.cuda())
        high_resolution_fake = g_net(Variable(low_resolution_real).cuda())
    else:
        high_resolution_real = Variable(high_resolution_real)
        high_resolution_fake = g_net(Variable(low_resolution_real))

    content_loss = 0
    adversarial_loss = 0

    # ----------------DISCRIMINATOR--------------

    discriminator_loss = criterion_2(d_net(high_resolution_real), ones_labels) + \
                         criterion_2(d_net(Variable(high_resolution_fake.data)), zeros_labels)


    # -----------------GENERATOR----------------

    tmp = vgg16cut(high_resolution_real)
    real = Variable(vgg16cut(high_resolution_real).data)
    fake = vgg16cut(high_resolution_fake)

    g_content_loss = criterion_1(high_resolution_fake, high_resolution_real) + 0.006 * criterion_1(fake, real)

    g_adversarial_loss = criterion_2(d_net(high_resolution_fake), ones_labels)

    total_loss = g_content_loss + 0.001 * g_adversarial_loss

    if i % 10 == 0:
        print('ciao')
