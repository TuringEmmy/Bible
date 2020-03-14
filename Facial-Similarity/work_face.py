#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   :-i https://pypi.douban.com/simple
# @Date     : 2020/3/2-13:34
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact  : WeChart：csy_lgy 
# @Project  : Bible-work_face
# @refere   :
# endregion
import torch
import torchvision

from torch import functional as F
from torch.autograd import Variable
from matplotlib import image as maping

from face import concatenated, imshow

img1 = "/Users/turing/Desktop/Bible/Facial-Similarity/data/testing/s5/1.pgm"
img2 = '/Users/turing/Desktop/Bible/Facial-Similarity/data/testing/s5/2.pgm'
img3 = "/Users/turing/Desktop/Bible/Facial-Similarity/data/testing/res/3.jpeg"
x0 = maping.imread(img1)
x1 = maping.imread(img2)
net = torch.load("best.ph")

x0 = torch.from_numpy(x0)
x1 = torch.from_numpy(x1)
# output1, output2 = net(Variable(x0), Variable(x1))
# euclidean_distance = F.pairwise_distance(output1, output2)
# print(euclidean_distance)
#
# folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
# siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
#                                         transform=transforms.Compose([transforms.Resize((100, 100)),
#                                                                       transforms.ToTensor()
#                                                                       ])
#                                         , should_invert=False)
#
# test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)
# dataiter = iter(test_dataloader)
# x0, _, _ = next(dataiter)


# _, x1, label2 = next(dataiter)
# concatenated = torch.cat((x0, x1), 0)

output1, output2 = net(Variable(x0), Variable(x1))
euclidean_distance = F.pairwise_distance(output1, output2)
imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))