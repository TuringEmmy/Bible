#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   :-i https://pypi.douban.com/simple
# @Date     : 2020/2/29-15:30
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact  : WeChart：csy_lgy 
# @Project  : Bible-binary_ops
# @refere   :
# endregion

from keras import backend as K


def round_through(x):
    """
    对x中的值取整数，同时使得求梯度的得到的值与原始的梯度一样
    小技巧：来自【Sergey Ioffe】(https://stackoverflow.com/a/36480182)
    :param x:
    :return:
    """
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmod(x):
    """
    当x<=-1, y=0;
    当-1<x<1,y=0.5*x+0.5
    当x>1,   y=1
    :param x:
    :return:
    """
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)  # clip :


def binary_tanh(x):
    """
    when forward propagation,output:
        x <= 0.0, y = -1
        x >  0.0, y = 1

    when backward propagation,
    rules as follows:
    2 * _hard_sigmod(x) -1

    x <= -1,   y = -1
    -1 < x <1, y =x
    x > 1      y = 1

    |x| > 1, 梯度为0
    :param x:
    :return:
    """
    return 2 * round_through(_hard_sigmod(x)) - 1


def binary(W, H=1):
    """
    二值化操作
    将[-H,H]之间的值转换为-H或者H
    :param W:
    :param H:
    :return:
    """
    Wb = H * binary_tanh(W / H)
    return Wb
