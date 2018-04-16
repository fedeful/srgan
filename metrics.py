import argparse, os
import torch
import torch.nn as nn
from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from models import SRGanGenerator

low_res_size = 32
ch_size = 3
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

parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


final_path = './weights'
dataset_folder = 'CelebA/'
test_dataset = dsets.ImageFolder(root=dataset_folder,
                                 transform=transforms.ToTensor())

train_sampler, valid_sampler, test_sampler = 3, 2, 5
# split_index_train_validation_test(len(train_dataset), 46, test_size, validation_size)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          sampler=train_sampler,
                                          shuffle=True,
                                          num_workers=2)


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
        raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

g_net = SRGanGenerator(16, 2)
g_pretrained_weights_dict = torch.load('%s/generator_final.pth' % (final_path))
g_net.load_state_dict(g_pretrained_weights_dict)

scales_factor = [2, 3, 4]

image_list = glob.glob(opt.dataset + "_mat/*.*")


for scale in scales_factor:
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0

    for i, data in enumerate(test_loader):
        low_resolution = torch.FloatTensor(1, ch_size, low_res_size, low_res_size)

        low_resolution = low_res_transform(high_resolution_real)
        high_resolution_real = norm(high_resolution_real)

        if cuda:
            high_resolution_real = Variable(high_resolution_real).cuda()
            high_resolution_fake = g_net(Variable(low_resolution).cuda())
        else:
            high_resolution_real = Variable(high_resolution_real)
            high_resolution_fake = g_net(Variable(low_resolution))

    for image_name in image_list:
        if str(scale) in image_name:
            count += 1
            print("Processing ", image_name)
            im_gt_y = sio.loadmat(image_name)['im_gt_y']
            im_b_y = sio.loadmat(image_name)['im_b_y']

            im_gt_y = im_gt_y.astype(float)
            im_b_y = im_b_y.astype(float)

            psnr_bicubic = PSNR(im_gt_y, im_b_y, shave_border=scale)
            avg_psnr_bicubic += psnr_bicubic

            im_input = im_b_y / 255.

            im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])

            if cuda:
                model = model.cuda()
                im_input = im_input.cuda()
            else:
                model = model.cpu()

            start_time = time.time()
            HR = model(im_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            HR = HR.cpu()

            im_h_y = HR.data[0].numpy().astype(np.float32)

            im_h_y = im_h_y * 255.
            im_h_y[im_h_y < 0] = 0
            im_h_y[im_h_y > 255.] = 255.
            im_h_y = im_h_y[0, :, :]

            psnr_predicted = PSNR(im_gt_y, im_h_y, shave_border=scale)
            avg_psnr_predicted += psnr_predicted

    print("Scale=", scale)
    print("Dataset=", opt.dataset)
    print("PSNR_predicted=", avg_psnr_predicted / count)
    print("PSNR_bicubic=", avg_psnr_bicubic / count)
    print("It takes average {}s for processing".format(avg_elapsed_time / count))