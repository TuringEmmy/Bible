#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2/4/20-12:24
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Image-Captioning-task2
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import listdir

from keras.models import model_from_json
from PIL import Image as pill_image
from keras import backend as K, Model
import numpy as np
from pickle import dump
from keras.models import Sequential
import keras


def load_img_as_np_array(path, target_size):
    """
    从给定的文件加载图像，转换图像大小为给定target_size,返回Keras支持的浮点数numpy数组
    :param path:图像文件路径
    :param target_size:元组（图像高度，图像宽度）
    :return:numpy数组
    """

    img = pill_image.open(path)
    img.resize(target_size, pill_image.NEAREST)

    return np.array(img, dtype=K.floatx())


def preprocess_input(x):
    """

    :param x:
    :return:
    """

    pass


def load_vgg16_model():
    """
    从当前目录下面的vgg16_exported.json和 vgg16_exported.h5两个文件导入VGG6网络并返回创建的网络模型
    :return: 创建的网络模型model
    """
    json_file = open('vgg16_exported.json', "r")
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weight("vgg16_exported.h5")
    return model


def extact_feature(directory):
    """
    提取给定文件夹中所有的图像特征，将提取的特征保存在文件features.pkl中，
    提取的特征保存在一个dict中，key为文件名（不带.jpg后缀）value为特征值[np.array]
    :param directory: 包含jpg文件的文件夹
    :return: None
    """
    model = load_vgg16_model()
    # 去掉模型的最后一层
    model.layers.pop()
    model = Model(inputs=model.inputs, output=model.layers[-1].output)

    features = dict()  # 定义一个pythion的数据字典
    for fn in listdir(directory):
        fn = directory + '/' + fn
        arr = load_img_as_np_array(fn, target_size=(224, 224))

        # 改变数组的形态，增加一个维度（批处理输入的维度）
        arr = arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))
        # 预处理图像作为VGG模型的输入
        arr = preprocess_input(arr)

        # 计算特赠

        feature = model.predict(arr, verbose=0)
        id = fn  # 去掉文件后缀
        features[id] = feature
