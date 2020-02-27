#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2/13/20-10:10
# @Motto    : Life is Short, I need use Python
# @Author   : turing
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Image-Captioning-task5
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def generate_caption(model, tokenizer, photo_feature, max_length=40):
    """
    根据输入的图像特征产生图像的标题
    :param model:预先训练好的图像标题生成神经网络模型
    :param tokenizer:一个预先产生的keras.preprocessing.text.Tokenizer
    :param photo_feature:输入的图像特征，为VGG16网络修改版产生的特征
    :param max_length:训练数据中最长的图像标题长度
    :return:产生的图像的标题(string)
    """


def word_for_id(integer, tokenizer):
    """
    将一个整数转换为英文单词
    :param integer: 一个代表英文的整数
    :param tokenizer: 一个预先产生的keras.preprocessing.Tokenizer
    :return: 输入整数对应的英文单词
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def evalute_model(model,captions,photo_features,tokenizer,max_length=40):
    """
    计算训练好的神经网络产生的标题
    :param model:训练好的产生标题的神经网络
    :param captions:测试数据集，key为文件名(不带.jpg后缀),value为图像标题list
    :param photo_features:dict,key文件名(不带.jpg后缀)，value为图像特征
    :param tokenizer:英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
    :param max_length:训练集中的标题的最大长度
    :return:
    """