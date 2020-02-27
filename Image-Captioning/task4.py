#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2/12/20-21:05
# @Motto    : Life is Short, I need use Python
# @Author   : turing
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Image-Captioning-task4
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from array import array
from json import load

from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

import utils


def create_batches(desc_list, photo_features, tokenizer, max_len, voc_size=7378):
    """
    从输入的图片标题list和图片特征构造LSTM的一组输入
    :param desc_list: 某一个图像对应的一组标题(一个list)
    :param photo_features: 某一个图像对应的特征
    :param tokenizer: 英文词和证书转换的工具keras.preprocessing.text.Tokenizer
    :param max_len: 训练数据集中最长的标题的长度
    :param voc_size: 训练集中的单词个数，默认为7378
    :return: 元组 1. list 图像的特征 2. list 图像标题的前缀 3. lsit图像标题的下一个单词(根据图像特征和标题的前缀产生)
    """
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequeue
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequece into multiple X,y pair
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
            out_seq = to_categorical([out_seq], num_classes=voc_size)[0]
            X1.append(photo_features)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


def data_generator(captions, photo_features, tokenizer, max_len):
    """
    创建一个训练数据生成器，用于传入模型训练函数的第一个参数model.fit_generator(generator,...)
    :param captions:dict,key为图像名(不包含.jpg后缀)，value为lsit,图像的几个训练标题
    :param photo_features:dict,key为图像名(不包含.jpg后缀)，value为图像的特征
    :param tokenizer:英文单词和证书转换的工具keras.preprocessing.text.Tokenizer
    :param max_len:
    :return:使用yied[[list(元素为图像特征)，list(元素为输入的图像的标题前缀)，list(元素为预期的输出图像标题的下一个单词)
    """
    # loop for ever images
    while 1:
        # retrieve the photo feature
        for key, desc_list in captions.items():
            photo_feature = photo_features[key][0]
            in_img, in_seq, out_word = create_batches(desc_list, photo_feature, tokenizer, max_len)
            yield [[in_seq], out_word]


def train():
    filename ='Flick_8k.trainImages.txt'
    train=utils.load_ids(filename)
    train_captions=utils.load_clean_captions('descriptions.txt',train)
    train_features=utils.load_photos_features('features.pkl',train)
    tokenizer = load(open('tokenizer.pkl','rb'))
    vocab_size = len(tokenizer.word_index)+1
    max_len = utils.get_max_length(train_captions)

    model = caption_model(vocab_size,max_len)
    epochs=20
    steps = len(train_captions)