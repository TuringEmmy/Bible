#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   :-i https://pypi.douban.com/simple
# @Date     : 2020/3/1-16:38
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact  : WeChart：csy_lgy 
# @Project  : Bible-face
# @refere   :
# endregion

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
import torchvision.datasets as dset
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from siamese_model import SiameseNetworkDataset, SiameseNetwork, ContrastiveLoss


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


class Config():
    """
    configuration class :a simple class to manage configution
    """
    train_dir = "data/training"
    testing_dir = "data/testing"
    train_batch_size = 4
    train_number_epochs = 100


# using image folder dataset
folder_dataset = dset.ImageFolder(root=Config.train_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose(
                                            [transforms.Resize((100, 100)), transforms.ToTensor()]),
                                        shouldInvert=False)
# visualising some of the data
# the top row and the bottom row of any column is one pair. The 0s and 1s correspond to the column of
# th eimage. 1 indiciates dissimilar, and 0 in dicates similar.
vis_dataloader = DataLoader(siamese_dataset, shuffle=True,
                            num_workers=2, batch_size=8)
dataiter = iter(vis_dataloader)

example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
# imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())

# train time
train_dataloader = DataLoader(siamese_dataset, shuffle=True,
                              num_workers=2, batch_size=Config.train_batch_size)
# net = SiameseNetwork().cuda()
net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)
counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        # img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print('epoch number {}\n current loss {}\n'.format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive)

show_plot(counter, loss_history)
