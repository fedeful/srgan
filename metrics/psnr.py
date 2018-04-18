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
import scipy.misc as misc
from skimage.measure import compare_psnr, compare_ssim, structural_similarity
from models import SRGanGenerator
from datasetloaders.rap_dataset import RAPDatasetTest, RAPDatasetTrainSRgan, RAPDatasetTestSRgan
import matplotlib.pyplot as plt

low_res_size = (328/4, 128/4)
high_res_size = (328, 128)

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


def psnr(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def from_torch_to_numpy(image):

    image = image.cpu().data[0]
    image = unormalization(image)
    image = image.numpy().astype(np.float32)
    image = image * 255.
    image[image < 0] = 0
    image[image > 255.] = 255.
    image = image.transpose((1, 2, 0)).astype(np.uint8)
    if True:
        plt.imshow(image)
        plt.show()

    return image


def psnr_evaluation(data_loader, g_net, model_path, cuda, scale_factor,ev_metrics, up_metrics):

    g_pretrained_weights_dict = torch.load('../%s/generator_ptrain.pth' % (model_path))
    g_net.load_state_dict(g_pretrained_weights_dict)
    g_net.eval()
    if cuda:
        g_net.cuda()

    avg_ssim_predicted = 0.
    avg_psnr_predicted = 0.
    avg_ssim_bicubic = 0.
    avg_psnr_bicubic = 0.
    avg_ssim_nearest = 0.
    avg_psnr_nearest = 0.
    count = 0.

    for i, data in enumerate(data_loader):

        count += 1
        high_resolution_real, low_resolution = data['imagehig'], data['imagelow']

        if cuda:
            low_resolution = Variable(low_resolution).cuda()
            high_resolution_real = Variable(high_resolution_real).cuda()
            high_resolution_fake = g_net(low_resolution)
        else:
            low_resolution = Variable(low_resolution)
            high_resolution_real = Variable(high_resolution_real)
            high_resolution_fake = g_net(Variable(low_resolution))

        low_resolution = from_torch_to_numpy(low_resolution)
        high_resolution_real = from_torch_to_numpy(high_resolution_real)
        high_resolution_fake = from_torch_to_numpy(high_resolution_fake)

        ssim_predicted = compare_ssim(high_resolution_real, high_resolution_fake, multichannel=True)
        psnr_predicted = compare_psnr(high_resolution_real, high_resolution_fake)
        avg_ssim_predicted += ssim_predicted
        avg_psnr_predicted += psnr_predicted

        bicubic = misc.imresize(low_resolution, size=high_res_size, interp='bicubic')
        ssim_bicubic = compare_ssim(high_resolution_real, bicubic, multichannel=True)
        psnr_bicubic = compare_psnr(high_resolution_real, bicubic)
        avg_ssim_bicubic += ssim_bicubic
        avg_psnr_bicubic += psnr_bicubic

        nearest = misc.imresize(low_resolution, size=high_res_size, interp='nearest')
        ssim_nearest = compare_ssim(high_resolution_real, nearest, multichannel=True)
        psnr_nearest = compare_psnr(high_resolution_real, nearest)
        avg_ssim_nearest += ssim_nearest
        avg_psnr_nearest += psnr_nearest

    return avg_ssim_predicted/count, avg_psnr_predicted/count, avg_ssim_bicubic/count, \
           avg_psnr_bicubic/count, avg_ssim_nearest/count, avg_psnr_nearest/count


if __name__ == '__main__':
    g_net = SRGanGenerator(16, 2)
    rap_folder = '../../../remote/datasets/'
    final_path = './weights'
    remember ='34,28'

    test_folder = RAPDatasetTestSRgan(rap_folder,
                                      rap_folder,
                                      pre_processing_hig,
                                      pre_processing_low)

    test_loader = torch.utils.data.DataLoader(dataset=test_folder,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=2,
                                              drop_last=True)



    a = psnr_evaluation(test_loader, g_net, final_path, True, 2)
    print(a)
