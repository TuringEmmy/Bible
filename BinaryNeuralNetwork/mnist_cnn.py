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
# @refere   : https://github.com/MatthieuCourbariaux/BinaryConnect/blob/lasagne/mnist.py
# https://github.com/Haosam/Binary-Neural-Network-Keras/blob/master/mnist_cnn.py
# endregion
# from keras.datasets import mnist
from keras import Sequential
from keras.layers import BatchNormalization, Activation, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils

from binary_layers import BinaryConv2D, BinaryDense
from binary_ops import binary_tanh as binary_tanh_op
from keras.callbacks import LearningRateScheduler

from data import get_mnist


def binary_tanh(x):
    return binary_tanh_op(x)


H = 1
kernel_lr_multiplier = 'Glorot'
# nn
batch_size = 50
epochs = 20
channels = 1
img_rows = 28
img_cols = 28
kernel_size = (3, 3)
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

# 下载MNIST数据集，。分为训练和测试数据
mnist = get_mnist()

(train_data, train_label, test_data, test_label) = mnist
# print(train_data.shape, train_label.shape)
train_data = train_data.reshape(60000, 1, 28, 28)
print(train_data.shape)
test_data = test_data.reshape(10000, 1, 28, 28)
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 225
test_data /= 225
print(train_data.shape[0], 'train samples')
print(test_data.shape[0], 'test samples')

# 将类别标签转为-1 或者1
train_label = np_utils.to_categorical(train_label, classes) * 2 - 1
test_label = np_utils.to_categorical(test_label, classes) * 2 - 1

model = Sequential()
model.add(BinaryConv2D(128, kernel_size=kernel_size, input_shape=(channels, img_rows, img_cols),
                       data_format='channels_first', H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='conv1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
model.add(Activation(binary_tanh, name='act1'))

model.add(BinaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first', padding='same', use_bias=use_bias, name='conv2'))
model.add(MaxPooling2D(pool_size=pool_size, data_format='channels_first', name='pool2'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
model.add(Activation(binary_tanh, name='act2'))

model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first', padding='same', use_bias=use_bias, name='conv3'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3'))
model.add(Activation(binary_tanh, name='act3'))

model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first', padding='same', use_bias=use_bias, name='conv4'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool4', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
model.add(Activation(binary_tanh, name='act4'))
model.add(Flatten())

# dense
model.add(BinaryDense(1024, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
model.add(Activation(binary_tanh, name='act5'))

model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

opt = Adam(lr=lr_start)
model.compile(loss='squared_hinge', optimizer=opt, metrics=['accuracy'])
model.summary()

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(train_data, train_label,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(test_data, test_label),
                    callbacks=[lr_scheduler])
score = model.evaluate(test_data, test_label, verbose=0)
# (test_data, test_label)
print('Test score:', score[0])
print("Test accuracy:", score[1])
