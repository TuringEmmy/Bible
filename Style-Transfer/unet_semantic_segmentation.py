#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2020/2/27-09:12
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Bible-unet_semantic_segmentation
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import cv2
from keras.layers import Conv2D, Activation, BatchNormalization, MaxPool2D, Dropout, Conv2DTranspose, concatenate
from keras.models import Model, Sequential
import tensorflow as tf


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """
    function to add 2 convolutional with the parameters to it
    :param input_tensor:
    :param n_filters:
    :param kernel_size:
    :param batchnorm:
    :return:
    """
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(x)
    if batchnorm:
        x = BatchNormalization(x)
    x = Activation('relu')(x)
    return x


def unet_model(input_img, n_filters=16, droput=0.1, batchnorm=True):
    """
    :param input_img:
    :param f_filters:
    :param droput:
    :param batchnorm:
    :return:
    """
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPool2D((2, 2))(c1)
    p1 = Dropout(droput)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPool2D((2, 2))(c2)
    p2 = Dropout(droput)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPool2D((2, 2))(c3)
    p3 = Dropout(droput)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPool2D((2, 2))(c4)
    p4 = Dropout(droput)(p4)

    c5 = conv2d_block(p4, n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    u6 = Conv2DTranspose(n_filters * 8, kernel_size=(3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(droput)(u6)
    u6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, kernel_size=3, strides=(2, 2), padding='same')(u6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(droput)(u7)
    u7 = conv2d_block(u7, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, kernel_size=(3, 3), strides=(2, 2), padding='same')(u7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(droput)(u8)
    u8 = conv2d_block(u8, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, kernel_size=3, strides=(2, 2), padding='same')(u8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(droput)(u9)
    u9 = conv2d_block(u9, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(filters=1, strides=(1, 1), activation='sigmoid')(u9)

    model = Model(inputs=[input_img], outputs=[outputs])
    with open('transfor_style.json', 'w') as f:
        f.write(json.dumps(json.loads(model.to_json()), indent=2))
    return model


if __name__ == '__main__':
    image = cv2.imread('data/IMG_7271.png')
    input_img = tf.convert_to_tensor(image,dtype=tf.float32,name='inputs_image')
    print(type(input_img))
    unet_model(input_img)
