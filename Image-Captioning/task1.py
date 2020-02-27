#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 1/31/20-12:49
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Image-Captioning-task1
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16

def generate_vgg16():
    """
    搭建VGG16网络结构
    :return: VGG16网络
    """
    input_shape = (224, 224, 3)
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2),strides=(2,2),),
        MaxPooling2D(pool_size=(2,2),strides=(2,2),),

        Flatten(),
        Dense(4096,activation='relu'),
        Dense(4096,activation='relu'),
        Dense(1000,activation='softmax'),

    ])

    return model



if __name__ == '__main__':
    model = generate_vgg16()
    model.summary()
