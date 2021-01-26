from __future__ import print_function
import os, ssl
import argparse
from math import log10
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from generators import ResnetGenerator, weights_init
from pytorch_wavelets import DWTForward, DWTInverse
import pytorch_ssim
import numpy as np
import math

# make sure we can download the pretrained model through proxy
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

torch.autograd.set_detect_anomaly(True)

# Training settings
parser = argparse.ArgumentParser(description='time-scale generative adversarial perturbations')
parser.add_argument('--imagenetTrain', type=str, default='/projects/jqiu_mlaas_projects/imageStore/public/imagenetData/imagenet/train', help='ImageNet train root')
parser.add_argument('--imagenetVal', type=str, default='/projects/jqiu_mlaas_projects/imageStore/public/imagenetData/imagenet/val', help='ImageNet val root')
parser.add_argument('--budget', type=float, default=0.1, help='budget for pertubation, choose from [0.1, 0.2, 0.3]')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: "adam" or "sgd"')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--MaxIter', type=int, default=500, help='Iterations in each Epoch')
parser.add_argument('--MaxIterTest', type=int, default=300, help='Iterations in each Epoch')
parser.add_argument('--expname', type=str, default='WGAP_Experiment', help='experiment name, output folder')
parser.add_argument('--checkpoint', type=str, default='', help='path to starting checkpoint')
parser.add_argument('--foolmodel', type=str, default='incv3', help='model to fool: "incv3", "vgg16", "vgg19", "resnet18", "resnet50" or "densenet161"')
parser.add_argument('--mode', type=str, default='train', help='mode: "train" or "test"')
parser.add_argument('--gpu_ids', help='gpu ids: e.g. 0 or 0,1 or 1,2.', type=str, default='0')
parser.add_argument('--wavelet_type', type=int, default=1, help='wavelet scale (1=1st wavelet; 2=2nd wavelet; 3=3rd wavelet')
opt = parser.parse_args()

# get wavelet type here
wavelet_type = opt.wavelet_type

print(opt)

if not torch.cuda.is_available():
   raise Exception("No GPU found.")

# train loss history
train_loss_history = []
test_loss_history = []
test_acc_history = []
test_fooling_history = []
best_fooling = 0
itr_accum = 0

# make directories
if not os.path.exists(opt.expname):
    os.mkdir(opt.expname)

cudnn.benchmark = True
torch.cuda.manual_seed(opt.seed)

MaxIter = opt.MaxIter
MaxIterTest = opt.MaxIterTest
gpulist = [int(i) for i in opt.gpu_ids.split(',')]
n_gpu = len(gpulist)
print('Running with n_gpu: ', n_gpu)

# define normalization means and stddevs
model_dimension = 299 if opt.foolmodel == 'incv3' else 256
center_crop = 299 if opt.foolmodel == 'incv3' else 224

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

print('===> Loading datasets')

if opt.mode == 'train':
    train_set = torchvision.datasets.ImageFolder(root = opt.imagenetTrain, transform = data_transform)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

test_set = torchvision.datasets.ImageFolder(root = opt.imagenetVal, transform = data_transform)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

# choose pretrained model, this is pretrained based ILSVRC2012
if opt.foolmodel == 'incv3':
    pretrained_clf = torchvision.models.inception_v3(pretrained=True)
    
elif opt.foolmodel == 'vgg16':
    pretrained_clf = torchvision.models.vgg16(pretrained=True)

elif opt.foolmodel == 'vgg19':
    pretrained_clf = torchvision.models.vgg19(pretrained=True)

elif opt.foolmodel == 'googlenet':
    pretrained_clf = torchvision.models.googlenet(pretrained=True)

elif opt.foolmodel == 'resnet18':
    pretrained_clf = torchvision.models.resnet18(pretrained=True)

elif opt.foolmodel == 'resnet50':
    pretrained_clf = torchvision.models.resnet50(pretrained=True)

elif opt.foolmodel == 'densenet161':
    pretrained_clf = torchvision.models.densenet161(pretrained=True)

pretrained_clf = pretrained_clf.cuda(gpulist[0])

pretrained_clf.eval()
pretrained_clf.volatile = True

print('===> Building model')

# will use model paralellism if more than one gpu specified
netG = ResnetGenerator(3*3, 3*3, opt.ngf, norm_type='batch', act_type='relu', gpu_ids=gpulist)

# resume from checkpoint if specified
if opt.checkpoint:
    if os.path.isfile(opt.checkpoint):
        print("=> loading checkpoint '{}'".format(opt.checkpoint))
        netG.load_state_dict(torch.load(opt.checkpoint, map_location=lambda storage, loc: storage))
        print("=> loaded checkpoint '{}'".format(opt.checkpoint))
    else:
        print("=> no checkpoint found at '{}'".format(opt.checkpoint))
        netG.apply(weights_init)
else:
    netG.apply(weights_init)

# setup optimizer
if opt.optimizer == 'adam':
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.optimizer == 'sgd':
    optimizerG = optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9)

criterion_pre = nn.CrossEntropyLoss()
criterion_pre = criterion_pre.cuda(gpulist[0])

# create forward and inverse wavelet transform
xfm = DWTForward(J=3, wave='db3', mode='symmetric').cuda(gpulist[0])
ifm = DWTInverse(wave='db3', mode='symmetric').cuda(gpulist[0])

# Penalty coef for conditional loss
dssimCoef = {
   0.05: 3.0,
   0.1: 2.0,
   0.2: 1.0,
   0.3: 0.5
}

#________________________________________________________________________________
#
## @brief conditional loss function with budget control
#
#________________________________________________________________________________
def conditionalBudgetLoss(perturbedX, x, outputLabel, targetLabel):
    # calculate cross entropy loss
    ce_criterion = nn.CrossEntropyLoss()
    ce_criterion.cuda(gpulist[0])
    CE = torch.log(ce_criterion(outputLabel, targetLabel))
    
    # calculate dssim
    ssim_loss = pytorch_ssim.SSIM()
    dssim = 1 - ssim_loss(perturbedX, x)

    # calculate l2 norm
    mse_loss = nn.MSELoss(reduction='mean')
    mse = mse_loss(perturbedX, x)
    l2Norm = math.sqrt(mse.item())

    if(l2Norm > opt.budget):
        return CE + dssimCoef[opt.budget]*dssim
    else:
        return CE

#________________________________________________________________________________
#
## @brief training method
#
#________________________________________________________________________________
def train(epoch):
    netG.train()
    global itr_accum
    global optimizerG

    # check for maximum number of iteration
    for itr, (image, _) in enumerate(training_data_loader, 1):
        if itr > MaxIter:
            break

        # least likely class in nontargeted case
        pretrained_label_float = pretrained_clf(image.cuda(gpulist[0]))
        _, target_label = torch.min(pretrained_label_float, 1)

        itr_accum += 1
        if opt.optimizer == 'sgd':
            lr_mult = (itr_accum // 1000) + 1
            optimizerG = optim.SGD(netG.parameters(), lr=opt.lr/lr_mult, momentum=0.9)

        image = image.cuda(gpulist[0])

        # decompose image wavelets
        Yl, Yh = xfm(image) 
        B, L, C, W, H = Yh[0].size()
        wavelet = Yh[0].clone()
        if wavelet_type == 2:
            B, L, C, W, H = Yh[1].size()
            wavelet = Yh[1].clone()
        elif wavelet_type == 3:
            B, L, C, W, H = Yh[2].size()
            wavelet = Yh[2].clone()

        ## generate per image perturbation from fixed noise
        input = wavelet.view((B,L*C,W,H))
        perturbedWavelet = netG(input)

        netG.zero_grad()
        
        if opt.foolmodel == 'incv3':
            if wavelet_type == 1:
                Yh[0]=perturbedWavelet.clone().view((B, L, C, W, H))
            elif wavelet_type == 2:       
                Yh[1]=nn.ConstantPad2d((0,-2,-2,0),0)(perturbedWavelet).clone().view((B, L, C, W, H))
            elif wavelet_type == 3:
                Yh[2]=nn.ConstantPad2d((0,-3,-3,0),0)(perturbedWavelet).clone().view((B, L, C, W, H))
        else:
            if wavelet_type == 1:
                Yh[0]=nn.ConstantPad2d((0,-2,-2,0),0)(perturbedWavelet).clone().view((B, L, C, W, H))
            elif wavelet_type == 2:       
                Yh[1]=nn.ConstantPad2d((0,-1,-1,0),0)(perturbedWavelet).clone().view((B, L, C, W, H))
            elif wavelet_type == 3:
                Yh[2]=perturbedWavelet.clone().view((B, L, C, W, H))

        # perform inverse wavelet transform to get perturbed images
        inverse = ifm((Yl, Yh))
        inverse = inverse.cuda(gpulist[0])

        if opt.foolmodel == 'incv3':
            # padding to make sure size match with original input
            recons = nn.ConstantPad2d((0,-1,-1,0),0)(inverse)   
        else:
            recons = inverse

        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image[:,cii,:,:].min(), image[:,cii,:,:].max())

        output_pretrained = pretrained_clf(recons.cuda(gpulist[0]))

        # attempt to get closer to least likely class, or target
        loss = conditionalBudgetLoss(recons.cuda(gpulist[0]), image.cuda(gpulist[0]), output_pretrained, target_label)

        loss.backward()
        optimizerG.step()

        train_loss_history.append(loss.item())
        print("===> Epoch[{}]({}/{}) loss: {:.4f}".format(epoch, itr, len(training_data_loader), loss.item()))

#________________________________________________________________________________
#
## @brief Testing method
#
#________________________________________________________________________________
def test():
    netG.eval()
    correct_recon = 0
    correct_orig = 0
    fooled = 0
    total = 0

    # check for maximum number of iteration
    for itr, (image, class_label) in enumerate(testing_data_loader):
        if itr > MaxIterTest:
            break

        # decompose images into wavelets and get the different scales
        image = image.cuda(gpulist[0])
        Yl, Yh = xfm(image) 
        B, L, C, W, H = Yh[0].size()
        wavelet = Yh[0].clone()
        if wavelet_type == 2:
            B, L, C, W, H = Yh[1].size()
            wavelet = Yh[1].clone()
        elif wavelet_type == 3:
            B, L, C, W, H = Yh[2].size()
            wavelet = Yh[2].clone()

        input = wavelet.view((B,L*C,W,H))
        perturbedWavelet = netG(input)

        if opt.foolmodel == 'incv3':
            if wavelet_type == 1:
                Yh[0]=perturbedWavelet.clone().view((B, L, C, W, H))
            elif wavelet_type == 2:       
                Yh[1]=nn.ConstantPad2d((0,-2,-2,0),0)(perturbedWavelet).clone().view((B, L, C, W, H))
            elif wavelet_type == 3:
                Yh[2]=nn.ConstantPad2d((0,-3,-3,0),0)(perturbedWavelet).clone().view((B, L, C, W, H))
        else:
            if wavelet_type == 1:
                Yh[0]=nn.ConstantPad2d((0,-2,-2,0),0)(perturbedWavelet).clone().view((B, L, C, W, H))
            elif wavelet_type == 2:       
                Yh[1]=nn.ConstantPad2d((0,-1,-1,0),0)(perturbedWavelet).clone().view((B, L, C, W, H))
            elif wavelet_type == 3:
                Yh[2]=perturbedWavelet.clone().view((B, L, C, W, H))

        # inverse wavelet transform to get images
        inverse = ifm((Yl, Yh))
        inverse = inverse.cuda(gpulist[0])
        
        if opt.foolmodel == 'incv3':
            # padding make sure size matches
            recons = nn.ConstantPad2d((0,-1,-1,0),0)(inverse)
        else:
            recons = inverse

        # get noise
        noise = torch.sub(recons.cuda(gpulist[0]), image.cuda(gpulist[0]))

        # do clamping per channel
        for cii in range(3):
            recons[:,cii,:,:] = recons[:,cii,:,:].clone().clamp(image[:,cii,:,:].min(), image[:,cii,:,:].max())

        outputs_recon = pretrained_clf(recons.cuda(gpulist[0]))
        outputs_orig = pretrained_clf(image.cuda(gpulist[0]))
        _, predicted_recon = torch.max(outputs_recon, 1)
        _, predicted_orig = torch.max(outputs_orig, 1)
        total += image.size(0)

        # calculate accuracy and fooling ratio
        correct_recon += (predicted_recon == class_label.cuda(gpulist[0])).sum()
        correct_orig += (predicted_orig == class_label.cuda(gpulist[0])).sum()
        fooled += (predicted_recon != predicted_orig).sum()

        if itr % 50 == 1:
            print('Images evaluated:', (itr*opt.testBatchSize))
            # undo normalize image color channels
            delta_im_temp = torch.zeros(noise.size())
            for c2 in range(3):
                recons[:,c2,:,:] = (recons[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                image[:,c2,:,:] = (image[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
                delta_im_temp[:,c2,:,:] = (noise[:,c2,:,:] * stddev_arr[c2]) + mean_arr[c2]
            if not os.path.exists(opt.expname):
                os.mkdir(opt.expname)

            torchvision.utils.save_image(recons, opt.expname+'/reconstructed_{}.png'.format(itr))
            torchvision.utils.save_image(image, opt.expname+'/original_{}.png'.format(itr))
            torchvision.utils.save_image(delta_im_temp, opt.expname+'/delta_im_{}.png'.format(itr))
            print('Saved images.')

    test_acc_history.append((100.0 * correct_recon / total))
    test_fooling_history.append((100.0 * fooled / total))
    print('Accuracy of the pretrained network on reconstructed images: %.2f%%' % (100.0 * float(correct_recon) / float(total)))
    print('Accuracy of the pretrained network on original images: %.2f%%' % (100.0 * float(correct_orig) / float(total)))
    print('Fooling ratio: %.2f%%' % (100.0 * float(fooled) / float(total)))

#________________________________________________________________________________
#
## @brief save checkpoint for specific epoch
#
#________________________________________________________________________________
def checkpoint_dict(epoch):
    netG.eval()
    global best_fooling
    if not os.path.exists(opt.expname):
        os.mkdir(opt.expname)

    net_g_model_out_path = opt.expname + "/netG_model_epoch_{}_".format(epoch) + "foolrat_{}.pth".format(test_fooling_history[epoch-1])
    if test_fooling_history[epoch-1] > best_fooling:
        best_fooling = test_fooling_history[epoch-1]
        torch.save(netG.state_dict(), net_g_model_out_path)
        print("Checkpoint saved to {}".format(net_g_model_out_path))
    else:
        print("No improvement:", test_fooling_history[epoch-1], "Best:", best_fooling)

#________________________________________________________________________________
#
## @brief pritn training/testing history
#
#________________________________________________________________________________
def print_history():
    # plot history for training loss
    if opt.mode == 'train':
        plt.plot(train_loss_history)
        plt.title('Model Training Loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.legend(['Training Loss'], loc='upper right')
        plt.savefig(opt.expname+'/reconstructed_loss_'+opt.mode+'.png')
        plt.clf()

    # plot history for classification testing accuracy and fooling ratio
    plt.plot(test_acc_history)
    plt.title('Model Testing Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Testing Classification Accuracy'], loc='upper right')
    plt.savefig(opt.expname+'/reconstructed_acc_'+opt.mode+'.png')
    plt.clf()

    plt.plot(test_fooling_history)
    plt.title('Model Testing Fooling Ratio')
    plt.ylabel('Fooling Ratio')
    plt.xlabel('Epoch')
    plt.legend(['Testing Fooling Ratio'], loc='upper right')
    plt.savefig(opt.expname+'/reconstructed_foolrat_'+opt.mode+'.png')
    print("Saved plots.")

if opt.mode == 'train':
    for epoch in range(1, opt.nEpochs + 1):
        print('Training...')
        train(epoch)
        print('Testing....')
        test()
        checkpoint_dict(epoch)
    print_history()
elif opt.mode == 'test':
    print('Testing...')
    test()
    print_history()
