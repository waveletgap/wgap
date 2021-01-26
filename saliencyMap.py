'''
generate saliency map for original and recon images on ineceptionV3
'''

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import math
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import ntpath
import glob
import os

# Captum model interpretability
from captum.attr import (
    Saliency,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
from captum.attr import visualization as viz

# transformation function
model_dimension = 299
center_crop = 299
mean_arr = [0.485, 0.456, 0.406]
stddev_arr = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean_arr,
                                 std=stddev_arr)
data_transform = transforms.Compose([
    transforms.Resize(model_dimension),
    transforms.CenterCrop(center_crop),
    transforms.ToTensor(),
    normalize,
])

# make computations deterministic
torch.manual_seed(123)
np.random.seed(123)

# get testing pretrained model
model = torchvision.models.inception_v3(pretrained=True)
model = model.cuda(0)
model.eval()

# parameters
parser = argparse.ArgumentParser(description='saliency map')
parser.add_argument('--original', type=str, default='', help='original image path')
parser.add_argument('--recon', type=str, default='', help='recon image path')
parser.add_argument('--output', type=str, default='', help='output folder')
opt = parser.parse_args()

def interpret_model(originalPath='',reconPath='', origOutput='', reconOutput=''):
    # read the images
    print("reading input image")
    original_image = cv.imread(originalPath)
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    recon_image = cv.imread(reconPath)
    recon_image = cv.cvtColor(recon_image, cv.COLOR_BGR2RGB)
    
    # creat torch tensor
    input = Image.open(originalPath)
    input = data_transform(input)
    input = torch.unsqueeze(input, 0)
    input.requires_grad = True
    recon = Image.open(reconPath)
    recon = data_transform(recon)
    recon = torch.unsqueeze(recon, 0)
    recon.requires_grad = True

    # do the classfication on the original image
    original_label_float = model(input.cuda(0))
    _, target_label = torch.max(original_label_float, 1)
    recon_label_float = model(recon.cuda(0))
    _, recon_label = torch.max(recon_label_float, 1)
    saliency = Saliency(model)
    grads = saliency.attribute(input.cuda(0), target = target_label)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    saliencyMap = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="all",
                            show_colorbar=True, title="Overlayed Saliency Map - Original")
    plt.savefig(origOutput + '/saliency_' + ntpath.basename(originalPath))

    grads = saliency.attribute(recon.cuda(0), target = recon_label)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    saliencyMap = viz.visualize_image_attr(grads, recon_image, method="blended_heat_map", sign="all",
                            show_colorbar=True, title="Overlayed Saliency Map - Recon")
    plt.savefig(reconOutput + '/saliency_' + ntpath.basename(reconPath))

originalImgList = glob.glob(opt.original + './*.png')
reconImgList = glob.glob(opt.recon + './*.png')

if not os.path.exists(opt.output):
        os.mkdir(opt.output)

reconOutput = opt.output + '/recon/'
originalOutput = opt.output + '/original/'

if not os.path.exists(reconOutput):
        os.mkdir(reconOutput)

if not os.path.exists(originalOutput):
        os.mkdir(originalOutput)

# loop through all image
for index, path in enumerate(originalImgList):
    print(str(index)+ ": Start processing image: " + str(path))
    interpret_model(originalImgList[index], reconImgList[index], originalOutput, reconOutput)
        