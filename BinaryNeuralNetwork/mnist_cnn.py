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
from keras import Sequential
from keras.layers import BatchNormalization, Activation, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.utils import np_utils

from binary_layers import BinaryConv2D, BinaryDense
from binary_ops import binary_tanh as binary_tanh_op
from keras.callbacks import LearningRateScheduler


def binary_tanh(x):
    return binary_tanh_op(x)


H = 1
kernel_lr_multiplier = 'Glorot'
# nn
batch_size = 50
epochs = 20
channels = 28
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


def get_mnist():
    # from sklearn.datasets import fetch_mldata
    # mnist = fetch_mldata('MNIST original', data_home="./")
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST', one_hot=True)
    return mnist


# 下载MNIST数据集，。分为训练和测试数据
mnist = get_mnist()
(X_train, y_train, X_test, y_test) = (mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels)
# print(X_train.shape, y_train.shape)
X_train = X_train.reshape(-1, 1, 28, 28)
# print(X_train.shape)
X_test = X_test.reshape(-1, 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 225
X_test /= 225
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 将类别标签转为-1 或者1
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)

model = Sequential()
model.add(BinaryConv2D(128, kernel_size=kernel_size, input_shape=(channels, img_rows, img_cols),
                       data_format='channels_first', H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       padding='same', use_bias=use_bias, name='conv1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
model.add(Activation(binary_tanh, nmae='act1'))

model.add(BinaryConv2D(128, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels_first', padding='same', use_bias=use_bias, name='conv2'))
model.add(MaxPooling2D(pool_size=pool_size, H=H, data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
model.add(Activation(binary_tanh, name='act2'))

model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels-first', padding='same', use_bias=use_bias, name='conv3'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3'))
model.add(Activation(binary_tanh, name='act3'))

model.add(BinaryConv2D(256, kernel_size=kernel_size, H=H, kernel_lr_multiplier=kernel_lr_multiplier,
                       data_format='channels-first', padding='same', use_bias=use_bias, name='conv4'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool4', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
model.add(Activation(binary_tanh, name='act4'))
model.add(Flatten())

# dense
model.add(BinaryDense(1024, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn5'))
model.add(Activation(binary_tanh, name='act5'))

model.add(BinaryConv2D(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense6'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn6'))

opt = Adam(lr=lr_start)
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print("Test accuracy:", score[1])
