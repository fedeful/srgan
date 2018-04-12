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
from printutils import print_partial_result

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


# MNIST Dataset
train_dataset = dsets.ImageFolder(root='./CelebA/',
                                  transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


transforms.Compose([
     transforms.CenterCrop(10),
     transforms.ToTensor(),
])

print_transform = transforms.Compose([transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
                                                           transforms.ToPILImage()])

scale_low_res = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(low_res_size),
                                   transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                   ])

norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

g_net = SRGanGenerator(5, 2)
d_net = SRGanDiscriminator(high_res_size)
g_optimizer = Adam(g_net.parameters(), lr=learning_rate)
d_optimizer = Adam(d_net.parameters(), lr=learning_rate)

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16cut = VggCutted(vgg16, 5)
criterion_1 = nn.MSELoss()
criterion_2 = nn.MSELoss()

if cuda:
    g_net.cuda()
    d_net.cuda()
    criterion_1.cuda()
    criterion_2.cuda()

b_fraction = len(train_dataset)/batch_size

'''

for epoch in np.arange(0, number_epochs):
    for i, data in enumerate(train_loader):

        images, labels = data

        high_resolution_real = images.clone()
        low_resolution_real = torch.FloatTensor(batch_size, ch_size, low_res_size, low_res_size)

        for j in np.arange(0, batch_size):
            low_resolution_real[j] = scale_low_res(high_resolution_real[j])
            high_resolution_real[j] = norm(high_resolution_real[j])

        if cuda:
            high_resolution_real = Variable(high_resolution_real.cuda())
            high_resolution_fake = g_net(Variable(low_resolution_real).cuda())
        else:
            high_resolution_real = Variable(high_resolution_real)
            high_resolution_fake = g_net(Variable(low_resolution_real))

        criterion_1.zero_grad()
        tmp = vgg16cut(high_resolution_real)

        # function(OUTPUT, TARGET)
        content_loss = criterion_1(high_resolution_fake, high_resolution_real)

        if i % 10 == 0:
            print('Iteration [%02d/%02d] Epoch [%02d/%02d] Generator content loss: %.5f' % (i, len(train_dataset)/batch_size, epoch, number_epochs, content_loss))
        content_loss.backward()
        if i % 10000 == 0:
            print_partial_result(low_resolution_real[0], high_resolution_real.data[0], high_resolution_fake.data[0],
                                print_transform)
        g_optimizer.step()

'''

ones_labels = Variable(torch.ones(batch_size, 1))
zeros_labels = Variable(torch.ones(batch_size, 1))

if cuda:
    ones_labels = ones_labels.cuda()
    zeros_labels = zeros_labels.cuda()


for epoch in np.arange(0, number_epochs):
    for i, data in enumerate(train_loader):

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

        #----------------DISCRIMINATOR--------------
        d_net.zero_grad()

        discriminator_loss = criterion_2(d_net(high_resolution_real), ones_labels) + \
                             criterion_2(d_net(Variable(high_resolution_fake.data)), zeros_labels)

        discriminator_loss.backward()
        d_optimizer.step()

        #-----------------GENERATOR----------------
        g_net.zero_grad()

        #real_features = Variable(feature_extractor(high_res_real).data)
        #fake_features = feature_extractor(high_res_fake)

        g_content_loss = criterion_1(high_resolution_fake, high_resolution_real)

        g_adversarial_loss = criterion_2(d_net(high_resolution_fake), ones_labels)

        total_loss = g_content_loss + 0.001 * g_adversarial_loss

        total_loss.backward()
        g_optimizer.step()

        if i % 10 == 0:
            print('Iteration [%02d/%02d] Epoch [%02d/%02d] Generator content loss: %02f Generator adverarial loss: %02f Discrimiator loss: 02%f' % \
                  (i, len(train_dataset)/batch_size, epoch, number_epochs, g_content_loss, g_adversarial_loss, discriminator_loss))


torch.save(g_net.state_dict(), '%s/generator_final.pth' % (final_path))
torch.save(d_net.state_dict(), '%s/discriminator_final.pth' % (final_path))



