#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2/12/20-15:50
# @Motto    : Life is Short, I need use Python
# @Author   : turing
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Image-Captioning-utils
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def load_image_nmaes(param):
    return None


def load_doc(filename):
    """
    读取文本文件为string
    :param filename: 文本文件
    :return: 文本文件的内容
    """
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def load_clean_captions(filename, dataset):
    """
    为图像标题首位分别加上'startseq'和'endseq',作为自动标题生成的其实和终止
    :param filename: 文本文件，每一行由图像名，和图像标题构成，图像的标题已经进行了清洗
    :param dataset: 图像名list
    :return: dict key为图像名，valuie为添加了'startseq'和'endseq'的标题的list
    """
    # load document
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        # split id from description
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list
            # wrap description in tokens
        desc = 'startseq' + ''.join(image_desc) + 'endseq'
        descriptions[image_id].append(desc)
    return descriptions


def to_list(caption):
    """
    将一个字典(key)为文件名，value为图像标题klist转换为图像表头list
    :param train_descriptions: 一个字典，key为文件名，vcalue为list
    :return: 图像标题list
    """
    all_desc = list()
    for key in caption.keys():
        [all_desc.append(d) for d in caption[key]]
    return all_desc
