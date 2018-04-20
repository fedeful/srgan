from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import Adam
from models import SRGanDiscriminator, SRGanGenerator, VggCutted
from logdatautils.printutils import NetworkInfoPrinter
from logdatautils.lossplot import save_loss_plot, save_2loss_plot
from torchvision.utils import save_image
from datasetloaders.rap_dataset import RAPDatasetTrainSRgan, RAPDatasetTestSRgan
import numpy as np

# ---------------PARAMETERS---------------#
beta = 0.006
lamb = 0.001
vgglayer = 5
number_epochs = 10
pretrain_epochs = 2
up_factor = 4
batch_size = 6
learning_rate = 0.0001
low_res_size = (328 / 4, 128 / 4)
high_res_size = (328, 128)
ch_size = 3
res_blocks = 16
up_blocks = 2
cutted_layer_vgg = 5
test_during_epoch = True
pre_train_flag = False
continue_adv_train = False
cuda = True

eval_path = '/home/federico/remote/projects/srgan/eval_plot'
weights_path = '/home/federico/remote/projects/srgan/weights'
printed_image_path = '/home/federico/remote/projects/srgan/printed_image'
logs_path = '/home/federico/remote/projects/srgan/logs'
# rap_folder = '../../datasets/'
rap_folder = '../../remote/datasets/'

# ---------------TRANSFORM----------------#

normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[1, 1, 1])

unormalization = transforms.Normalize(mean=[-0.5, -0.5, -0.5],
                                      std=[1, 1, 1])

pre_processing_hig = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(high_res_size),
                                         transforms.ToTensor(),
                                         normalization])

pre_processing_low = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(low_res_size),
                                         transforms.ToTensor(),
                                         normalization])

# -------------DATA-LOADER---------------#

train_folder = RAPDatasetTrainSRgan(rap_folder,
                                    rap_folder,
                                    pre_processing_hig,
                                    pre_processing_low)

test_folder = RAPDatasetTestSRgan(rap_folder,
                                  rap_folder,
                                  pre_processing_hig,
                                  pre_processing_low)

train_loader = torch.utils.data.DataLoader(dataset=train_folder,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_folder,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2,
                                          drop_last=True)

# -------------NETWORK-SETUP----------------#

g_net = SRGanGenerator(res_blocks, up_factor / 2)
d_net = SRGanDiscriminator()

print(g_net)
print(d_net)

g_optimizer = Adam(g_net.parameters(), lr=learning_rate)
d_optimizer = Adam(d_net.parameters(), lr=learning_rate)

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16cut = VggCutted(vgg16, vgglayer)

criterion_1 = nn.MSELoss()
criterion_2 = nn.BCELoss()

b_fraction = len(train_folder) / batch_size

ones_labels = Variable(torch.ones(batch_size, 1))
zeros_labels = Variable(torch.zeros(batch_size, 1))

if cuda:
    g_net.cuda()
    d_net.cuda()
    criterion_1.cuda()
    criterion_2.cuda()
    vgg16cut.cuda()
    ones_labels = ones_labels.cuda()
    zeros_labels = zeros_labels.cuda()


def generator_pre_training(epoch, iter_print_loss, iter_print_image):
    g_net.train()

    nips.title_line('GENERAOTR-PRE-TRAINING')

    total_round = 0.
    avg_g_content_loss = 0.

    for i, data in enumerate(train_loader):

        total_round += 1

        high_resolution_real, low_resolution = data['imagehig'], data['imagelow']

        if cuda:
            high_resolution_real = Variable(high_resolution_real).cuda()
            high_resolution_fake = g_net(Variable(low_resolution).cuda())
        else:
            high_resolution_real = Variable(high_resolution_real)
            high_resolution_fake = g_net(Variable(low_resolution))

        g_net.zero_grad()
        content_loss = criterion_1(high_resolution_fake, high_resolution_real)
        avg_g_content_loss += content_loss.data[0]
        content_loss.backward()
        g_optimizer.step()

        if i % iter_print_loss == 0:
            tmp_dic = dict()
            tmp_dic['Generator Loss'] = ('%s: %.7f', content_loss)
            nips.log_line(epoch, i, tmp_dic)
            nipf.log_line(epoch, i, tmp_dic)

        if i % iter_print_image == 0 and iter_print_image != 0:
            save_image(unormalization(high_resolution_real.data[0]),
                       '%s/high_resolution_real/pre_train_%s_%s.png' % (printed_image_path, epoch, i))
            save_image(unormalization(high_resolution_fake.data[0]),
                       '%s/high_resolution_fake/pre_train_%s_%s.png' % (printed_image_path, epoch, i))
            save_image(unormalization(low_resolution[0]),
                       '%s/low_resolution/pre_train_%s_%s.png' % (printed_image_path, epoch, i))

    return avg_g_content_loss / total_round


def adversarial_training(epoch, iter_print_loss, iter_print_image):
    nips.title_line('TRAINING')
    g_net.train()
    d_net.train()

    total_round = 0.
    avg_d_loss = 0.
    avg_g_content_loss = 0.
    avg_g_adv_loss = 0.
    avg_g_tot_loss = 0.

    for i, data in enumerate(train_loader):

        high_resolution_real, low_resolution = data['imagehig'], data['imagelow']

        real_labels = Variable(torch.rand(batch_size, 1) * 0.5 + 0.75)
        fake_labels = Variable(torch.rand(batch_size, 1) * 0.25)

        if cuda:
            fake_labels = fake_labels.cuda()
            real_labels = real_labels.cuda()
            high_resolution_real = Variable(high_resolution_real).cuda()
            high_resolution_fake = g_net(Variable(low_resolution).cuda())
        else:
            high_resolution_real = Variable(high_resolution_real)
            high_resolution_fake = g_net(Variable(low_resolution))

        # ----------------DISCRIMINATOR--------------#

        d_net.zero_grad()
        d_loss = criterion_2(d_net(high_resolution_real), real_labels) + criterion_2(
            d_net(Variable(high_resolution_fake.data)), fake_labels)
        avg_d_loss += d_loss.data[0]
        d_loss.backward()
        d_optimizer.step()

        # -----------------GENERATOR----------------#

        g_net.zero_grad()
        real = Variable(vgg16cut(high_resolution_real).data)
        fake = vgg16cut(high_resolution_fake)

        g_content_loss_1 = criterion_1(high_resolution_fake, high_resolution_real)
        g_content_loss_2 = criterion_1(fake, real)
        g_content_loss = g_content_loss_1 + beta * g_content_loss_2
        avg_g_content_loss += g_content_loss.data[0]

        g_adversarial_loss = criterion_2(d_net(high_resolution_fake), ones_labels)
        avg_g_adv_loss += g_adversarial_loss.data[0]

        g_total_loss = g_content_loss + lamb * g_adversarial_loss
        avg_g_tot_loss += g_total_loss.data[0]

        g_total_loss.backward()
        g_optimizer.step()

        total_round += 1

        if i % iter_print_loss == 0:
            tmp_dic = dict()
            tmp_dic['Discriminator Loss'] = ('%s: %.7f ', d_loss)
            tmp_dic['Generator Content Loss'] = ('%s: %.7f ', g_content_loss)
            tmp_dic['Generator Adversarial Loss'] = ('%s: %.7f ', g_adversarial_loss)
            tmp_dic['Generator Total Loss'] = ('%s: %.7f ', g_total_loss)
            nips.log_line(epoch, i, tmp_dic)
            nipftr.log_line(epoch, i, tmp_dic)

        if i % iter_print_image == 0 and iter_print_image != 0:
            save_image(high_resolution_real.data[0],
                       '%s/high_resolution_real/adverse_%s_%s.png' % (printed_image_path, epoch, i))
            save_image(high_resolution_fake.data[0],
                       '%s/high_resolution_fake/adverse_%s_%s.png' % (printed_image_path, epoch, i))
            save_image(low_resolution[0],
                       '%s/low_resolution/adverse_%s_%s.png' % (printed_image_path, epoch, i))

    return avg_d_loss / total_round, avg_g_content_loss / total_round, avg_g_adv_loss / total_round, avg_g_tot_loss / total_round


def testing(epoch, iter_print_loss, iter_print_image):
    nips.title_line('TEST')
    g_net.eval()
    d_net.eval()

    total_round = 0.
    avg_d_loss = 0.
    avg_g_content_loss = 0.
    avg_g_adv_loss = 0.
    avg_g_tot_loss = 0.

    for i, data in enumerate(test_loader):

        high_resolution_real, low_resolution = data['imagehig'], data['imagelow']

        if cuda:
            high_resolution_real = Variable(high_resolution_real.cuda(), volatile=True)
            high_resolution_fake = g_net(Variable(low_resolution, volatile=True).cuda())

        else:
            high_resolution_real = Variable(high_resolution_real, volatile=True)
            high_resolution_fake = g_net(Variable(low_resolution, volatile=True))

        # ----------------DISCRIMINATOR--------------

        d_loss = (criterion_2(d_net(high_resolution_real), ones_labels) +
                  criterion_2(d_net(Variable(high_resolution_fake.data, volatile=True)), zeros_labels)).data[0]
        avg_d_loss += d_loss

        # -----------------GENERATOR----------------

        real = Variable(vgg16cut(high_resolution_real).data, volatile=True)
        fake = vgg16cut(high_resolution_fake)

        gcl = (criterion_1(high_resolution_fake, high_resolution_real) + beta * criterion_1(fake, real)).data[0]
        avg_g_content_loss += gcl

        gal = criterion_2(d_net(high_resolution_fake), ones_labels).data[0]
        avg_g_adv_loss += gal

        g_total_loss = gcl + lamb * gal
        avg_g_tot_loss += g_total_loss

        if i % iter_print_loss == 0:
            tmp_dic = dict()
            tmp_dic['Discriminator Loss'] = ('%s: %.7f ', d_loss)
            tmp_dic['Generator Content Loss'] = ('%s: %.7f ', gcl)
            tmp_dic['Generator Adversarial Loss'] = ('%s: %.7f ', gal)
            tmp_dic['Generator Total Loss'] = ('%s: %.7f ', g_total_loss)
            nips.log_line(epoch, i, tmp_dic)
            nipfts.log_line(epoch, i, tmp_dic)

        if i % iter_print_image == 0 and iter_print_image != 0:
            save_image(high_resolution_real.data[0],
                       '%s/high_resolution_real/test_%s_%s.png' % (printed_image_path, epoch, i))
            save_image(high_resolution_fake.data[0],
                       '%s/high_resolution_fake/test_%s_%s.png' % (printed_image_path, epoch, i))
            save_image(low_resolution[0],
                       '%s/low_resolution/test_%s_%s.png' % (printed_image_path, epoch, i))

        total_round += 1

    return avg_d_loss / total_round, avg_g_content_loss / total_round, avg_g_adv_loss / total_round, avg_g_tot_loss / total_round


# ---------------GENERATOR-PRE-TRAINING--------------#


nips = NetworkInfoPrinter('', number_epochs, len(train_folder), batch_size)

if pre_train_flag:
    nipf = NetworkInfoPrinter('%s/g_pretrain.txt' % logs_path, number_epochs, len(train_folder), batch_size)
    gen_loss = []

    for epoch_n in np.arange(0, pretrain_epochs):
        gen_loss.append(generator_pre_training(epoch_n, 50, 500))

    save_loss_plot('%s/pre_train_loss.png' % eval_path, 'Pre Train loss', np.arange(0, pretrain_epochs), gen_loss)
    nipf.end_print()
    torch.save(g_net.state_dict(), '%s/generator_ptrain.pth' % weights_path)
else:
    if continue_adv_train:
        print('ciao')
    else:
        g_pretrained_weights_dict = torch.load('%s/generator_ptrain.pth' % weights_path)
        g_net.load_state_dict(g_pretrained_weights_dict)

# ---------------ADVERSARIAL-TRAINING--------------#


nipftr = NetworkInfoPrinter('%s/advers_train.txt' % logs_path, number_epochs, len(train_folder), batch_size)
nipfts = NetworkInfoPrinter('%s/advers_test.txt' % logs_path, number_epochs, len(test_folder), batch_size)

d_loss_ltr = []
g_content_loss_ltr = []
g_adv_loss_ltr = []
g_tot_loss_ltr = []

d_loss_lts = []
g_content_loss_lts = []
g_adv_loss_lts = []
g_tot_loss_lts = []

for epoch_n in np.arange(0, number_epochs):

    loss_train = adversarial_training(epoch_n, 50, 2000)
    if test_during_epoch:
        loss_test = testing(epoch_n, 50, 2000)

    d_loss_ltr.append(loss_train[0])
    g_content_loss_ltr.append(loss_train[1])
    g_adv_loss_ltr.append(loss_train[2])
    g_tot_loss_ltr.append(loss_train[3])

    if test_during_epoch:
        d_loss_lts.append(loss_test[0])
        g_content_loss_lts.append(loss_test[1])
        g_adv_loss_lts.append(loss_test[2])
        g_tot_loss_lts.append(loss_test[3])

if test_during_epoch:
    save_2loss_plot('%s/adv_disc_loss.png' % eval_path, 'Discriminator loss', ('Train', 'Test'),
                    np.arange(0, number_epochs),
                    d_loss_ltr, d_loss_lts)
    save_2loss_plot('%s/adv_gen_loss.png' % eval_path, 'Generator loss', ('Train', 'Test'), np.arange(0, number_epochs),
                    g_tot_loss_ltr, g_tot_loss_lts)
else:
    save_loss_plot('%s/adv_disc_loss.png' % eval_path, 'Discriminator loss',
                   np.arange(0, number_epochs),
                   d_loss_ltr)
    save_loss_plot('%s/adv_gen_loss.png' % eval_path, 'Generator loss',
                   np.arange(0, number_epochs), g_tot_loss_ltr)

nipftr.end_print()
nipfts.end_print()
nips.end_print()

torch.save(g_net.state_dict(), '%s/generator_final.pth' % weights_path)
torch.save(d_net.state_dict(), '%s/discriminator_final.pth' % weights_path)
