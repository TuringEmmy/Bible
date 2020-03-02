#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   :-i https://pypi.douban.com/simple
# @Date     : 2020/3/1-17:03
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact  : WeChart：csy_lgy 
# @Project  : Bible-model
# @refere   :
# endregion
import random

import torch
from PIL import ImageOps, Image
from torch.utils.data import Dataset
import numpy as np

from torch.nn import Module, Sequential, ReflectionPad2d, Conv2d, ReLU, BatchNorm2d, Linear
from torch.nn import functional as F

from torch import nn


class SiameseNetworkDataset(Dataset):
    """
    this dataset generates a pair of images.  0 for geniune pair an d 1 for imposter pair
    """

    def __init__(self, imageFolderDataset, transform=None, shouldInvert=True):
        self.imageFoldDataset = imageFolderDataset
        self.transform = transform
        self.shouleInvert = shouldInvert

    def __getitem__(self, item):
        img0_tuple = random.choice(self.imageFoldDataset.imgs)
        # we need to make sure approx 50% of images are in the same class

        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep loopin till the same class is found

                img1_tuple = random.choice(self.imageFoldDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break

        else:
            while True:
                # keep looping till a different class image is found
                img1_tuple = random.choice(self.imageFoldDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        # 转换成黑白
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.shouleInvert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        result = torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))  # tensor
        return img0, img1, result

    def __len__(self):
        return len(self.imageFoldDataset.imgs)


class SiameseNetwork(Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = Sequential(
            ReflectionPad2d(1),
            Conv2d(1, 4, kernel_size=3),
            ReLU(inplace=True),
            BatchNorm2d(4),

            ReflectionPad2d(1),
            Conv2d(4, 8, kernel_size=3),
            ReLU(inplace=True),
            BatchNorm2d(8),

            ReflectionPad2d(1),
            Conv2d(8, 8, kernel_size=3),
            ReLU(inplace=True),
            BatchNorm2d(8),
        )

        self.fc1 = Sequential(
            Linear(8 * 100 * 100, 500),
            ReLU(inplace=True),

            Linear(500, 500),
            ReLU(inplace=True),

            Linear(500, 5)
        )

        def forward_once(self, x):
            output = self.cnn1(x)
            output = output.view(output.size()[0], -1)
            output = self.fc1(output)
            return output

        def forward(self, input1, input2):
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            return output1, output2


class SiameseNetwork1(nn.Module):
    def __init__(self):
        super(SiameseNetwork1, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(Module):
    """
    based on: https://yan.lecun.com/exdb/publis/hadsell-chopra-lecun-06.pdf
              http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, outputs1, outputs2, label):
        euclidean_distance = F.pairwise_distance(outputs1, outputs2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
