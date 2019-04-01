#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:29:53 2019

@author: bryan
"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image

from torchvision import transforms
import argparse

import Hash_functions as H_F
import DataProcessing as DP

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='data/CIFAR-10', help='cifar10 | nus-wide | mirflickr-25k')
parser.add_argument('--database', default='database_img.txt', help='cifar10 | nus-wide | mirflickr-25k')
parser.add_argument('--test_file', default='test_img.txt', help='cifar10 | nus-wide | mirflickr-25k')
parser.add_argument('--database_label', default='database_label.txt', help='cifar10 | nus-wide | mirflickr-25k')
parser.add_argument('--test_label',  default='test_label.txt', help='cifar10 | nus-wide | mirflickr-25k')
parser.add_argument('--model_name', default='cnn-f', help='the height / width of the input image to network')
parser.add_argument('--use_gpu', default=True, help='size of the latent z vector')
parser.add_argument('--epoch', default=1000, help='size of the latent z vector')
parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--imagesize', type=int, default=224, help='the height / width of the input image to hash network')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--nclass', type=int, help='number of data loading workers', default=10)
parser.add_argument('--gpuid', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--bits', type=int, default=128, help='number of GPUs to use')
parser.add_argument('--index', type=int, default=4, help='the index of test images')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
opt = parser.parse_args()
print(opt)

transformations = transforms.Compose([
        transforms.Resize(opt.imagesize),
        transforms.CenterCrop(opt.imagesize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

print('pre-process data...')
dset_test = DP.DatasetProcessingCIFAR_10(opt.dataroot, opt.test_file, opt.test_label, transformations)

print('loading test data...')
num_test = len(dset_test)
test_loader = DataLoader(dset_test, batch_size=opt.batchsize, shuffle=False, num_workers=4)

print('loading trained hash network...')
model = torch.load('./model/netF_128bits.pth')
model.eval()

print('loading trained inverse hash network...')
netG = torch.load('./model/generator.pth')
netG = netG.cuda()

print('generate test data into hash codes...')
qB = H_F.Generate_hash_code(model, test_loader, num_test, opt.bits, opt.use_gpu)

def reconstruct_images(index, netG):
    insert_code = qB[index * opt.batchsize + 1: (index + 1) * opt.batchsize + 1]
    insert_code = torch.Tensor(insert_code)
    insert_code = Variable(insert_code.float()).cuda()
    fake_images = netG(insert_code)
    
    return fake_images

def save_image(filename, image):
    img = image.data.add(1).div(2).mul(255).clamp(0, 255).cpu().numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

reconstructive_images = reconstruct_images(4, netG)
for i in range(opt.batchsize-1):
    print('save %d-th reconstructive image' %i)
    save_image('./reconstruction/generated_image_%d.png' %i, reconstructive_images[i])

print('Testing is complete! The results are saveed in the folder of <reconstruction>')
print('-----------------------------------------------------------------------------')
