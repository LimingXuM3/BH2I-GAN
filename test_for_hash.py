from torchvision import transforms
import torch
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import Hash_functions as H_F
import Load_functions as L_F
import utils.DataProcessing as DP
import utils.CalcHammingRanking as CalcHR

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='data/CIFAR-10', help='cifar10 | coco | mirflickr-25k')
parser.add_argument('--database', default='database_img.txt', help='cifar10 | coco | mirflickr-25k')
parser.add_argument('--test_file', default='test_img.txt', help='cifar10 | coco | mirflickr-25k')
parser.add_argument('--database_label', default='database_label.txt', help='cifar10 | coco | mirflickr-25k')
parser.add_argument('--test_label',  default='test_label.txt', help='cifar10 | coco | mirflickr-25k')
parser.add_argument('--model_name', default='cnn-f', help='the pre-trained hash network')
parser.add_argument('--use_gpu', default=True, help='Whether use GPU')
parser.add_argument('--batchsize', type=int, default=128, help='input batch size')
parser.add_argument('--imagesize', type=int, default=224, help='the height / width of the input image to hash network')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nclass', type=int, help='the classes of image dataset', default=10)
parser.add_argument('--gpuid', type=int, default=0, help='The id of GPU')
parser.add_argument('--bits', type=int, default=16, help='The length of hash codes')
parser.add_argument('--lamda', type=int, default=50, help='hyper parameter, default=50')
opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuid)

transformations = transforms.Compose([
        transforms.Resize(opt.imagesize),
        transforms.CenterCrop(opt.imagesize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

print('pre-process data...')
dset_database = DP.DatasetProcessingCIFAR_10(opt.dataroot, opt.database, opt.database_label, transformations)
dset_test = DP.DatasetProcessingCIFAR_10(opt.dataroot, opt.test_file, opt.test_label, transformations)

print('loading test data...')
num_database, num_test = len(dset_database), len(dset_test)
database_loader = DataLoader(dset_database, batch_size=opt.batchsize, shuffle=False, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=opt.batchsize, shuffle=False, num_workers=4)

print('loading test label...')
test_labels = L_F.Load_label(opt.test_label, opt.dataroot)
test_labels_onehot = H_F.Encode_onehot(test_labels, opt.nclass)

print('loading trained model...')
model = torch.load('./model/netF_16bits.pth')
model.eval()

database_labels = L_F.Load_label(opt.database_label, opt.dataroot)
database_labels_onehot = H_F.Encode_onehot(database_labels, opt.nclass)

print('generate test data into hash codes...')
qB = H_F.Generate_hash_code(model, test_loader, num_test, opt.bits, opt.use_gpu)
dB = H_F.Generate_hash_code(model, database_loader, num_database, opt.bits, opt.use_gpu)

print('Calculating MAP...')
MAP = CalcHR.CalcMap(qB, dB, test_labels_onehot.numpy(), database_labels_onehot.numpy())
print('MAP(retrieval database): %3.5f' % MAP)
