#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   :-i https://pypi.douban.com/simple
# @Date     : 2020/2/29-17:21
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact  : WeChart：csy_lgy 
# @Project  : Bible-mnist_cnn
# @refere   :
# endregion
# from keras.datasets import mnist

from binary_ops import binary_tanh as binary_tanh_op


def binary_tanh(x):
    return binary_tanh_op(x)


H = 1
kernel_lr_multiplier = 'Glorot'
# nn
batch_size = 50
epochs = 20
channels = 28
img_rows = 28
mg_cols = 28
kerel_size = (3, 3)
pool_size = (2, 2)
classes = 10
use_bias = False

# 学习率变化安排
lr_start = 1e-3
le_end = 1e-4
lr_decay = (le_end / lr_start) ** (1 / epochs)

# Batch Normalization 参数

epsilon = 1e-6
momentum = 0.9


def get_mnist():
    # from sklearn.datasets import fetch_mldata
    # mnist = fetch_mldata('MNIST original', data_home="./")
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST', one_hot=True)
    return mnist


# 下载MNIST数据集，。分为训练和测试数据
mnist = get_mnist()
(X_train, y_train, X_test, y_test) = (mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels)
print(X_test)
