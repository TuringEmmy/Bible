#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2/4/20-13:30
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Image-Captioning-task3
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import utils


def create_tokenizer():
    """
    根据训练数据集中图像名，和其它对应的标题，生成一个tokenizer
    :return: 生成的tokenizer
    """
    train_image_names = utils.load_image_nmaes('Flickr_8k.trainTmages.txt')
    train_descriptions = utils.load_clean_captions('descriptions.txt', train_image_names)
    lines = utils.to_list(train_descriptions)


def create_input_data_for_one_image():
    pass
