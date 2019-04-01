# BH2I-GAN

# Overview

A test implementation for the paper "BH2I-GAN: Bidirectional Hash code-to-Image Translation usingMulti-Generative Multi-Adversaria" (Under reviewing)

# Environment: 
  python 3.6

# Supported Toolkits
  pytorch (Pytorch http://pytorch.org/)
  
  torchvision
  
  numpy
  
# Demo

  1. Download pre-trained models from [BaiduNetdisk](https://pan.baidu.com/s/1z8lhFlAr3_YthTrNywMNtw). password: amwa.

  2. Download test samples from [BaiduNetdisk](https://pan.baidu.com/s/1z8lhFlAr3_YthTrNywMNtw), then put all this data into corresponding dir and extract compressed files.
       
  3. Copy the model into your dir
  
     cp netG*.pth ./model/
     
     cp netF*.pth ./model/

  4. Test for the retrieval performance of proposed BH2I-GAN
  
     python test_for_hash.py
     
  5. Test for the reconstructive performance of proposed BH2I-GAN
  
     python test_for_inverse_hash.py
        
# Notes
- This is developed on a Linux machine running Ubuntu 16.04.
- Use GPU for the high speed computation.
