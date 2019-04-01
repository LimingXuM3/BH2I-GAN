#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 11:08:19 2018

@author: bryan
"""

import torch
from torch.autograd import Variable
import numpy as np
import BH2I_GAN_models
from torchvision import models

def CreateModel(model_name, bit, use_gpu):
    if model_name == 'vgg19':
        vgg19 = models.vgg19(pretrained=True)
        hashnet = BH2I_GAN_models.Hash_model(vgg19, model_name, bit)
    if model_name == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
        hashnet = BH2I_GAN_models.Hash_model(alexnet, model_name, bit)
    if use_gpu:
        hashnet = hashnet.cuda()
    return hashnet

def Encode_onehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

def Calc_Sim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return S

def Calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def Calc_Supvervised_manifold_Sim(batch_label, train_label):
    S_pa = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    S_m = batch_label * S_pa
    return S_m

def Inner_product(code1, code2):
    inner = np.dot(code1, code2.transpose())
    inner = np.array(inner, dtype=np.float32)
    
    return inner

def Totloss(U, B, Sim, lamda, num_train):
    theta = U.mm(U.t()) / 2
    t1 = (theta*theta).sum() / (num_train * num_train)
    l1 = (- theta * Sim + Logtrick(Variable(theta), False).data).sum()
    l2 = (U - B).pow(2).sum()
    l = l1 + lamda * l2
    return l, l1, l2, t1

def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.])))
    return lt

def W_balance(source_array):
    count = 0
    w_ij = 0
    raw = source_array.shape[0]
    col = source_array.shape[1]
    num_s = float(raw * col)
    for i in range(raw):
        for x in source_array[i]:
            if x == 1:
                count += 1
    w_ij = num_s / count
    
    return w_ij

def Weights_init(m):  # custom weights initialization called on netG and netD
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def Calc_map(qB, rB, queryL, retrievalL):
    num_query = queryL.shape[0]
    map = 0

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = Calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
    map = map / num_query

    return map

def Calc_topmap(qB, rB, queryL, retrievalL, topk):
    num_query = queryL.shape[0]
    topkmap = 0

    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = Calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query

    return topkmap

def LearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def Generate_hash_code(model, data_loader, num_data, bit, use_gpu):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda())
        else: data_input = Variable(data_input)
        output = model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B
