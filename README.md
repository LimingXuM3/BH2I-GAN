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

  1. Download pre-trained models from [BaiduNetdisk](https://pan.baidu.com/s/1f6XUL8QVIO5C9j6pGEwhzw). password: k22r.

  2. Download dataset from [BaiduNetdisk](https://pan.baidu.com/s/1ng-zBf6vlVrCkNe4ObnW6w). password: 7ugb. Then put all the data into corresponding dir and extract compressed files.
       
  3. Copy the model into your dir
  
     cp hash_network*.pth ./model/
     
     cp inverse_hash_network.pth ./model/

  4. Test for the retrieval performance of proposed BH2I-GAN
  
     python test_for_hash.py
     
  5. Test for the reconstructive performance of proposed BH2I-GAN
  
     python test_for_inverse_hash.py
        
# Notes
- This is developed on a Linux machine running Ubuntu 16.04.
- Use GPU for the high speed computation.
