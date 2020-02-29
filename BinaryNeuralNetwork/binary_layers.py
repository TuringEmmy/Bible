#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   :-i https://pypi.douban.com/simple
# @Date     : 2020/2/29-15:29
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact  : WeChart：csy_lgy 
# @Project  : Bible-binary_layers
# @refere   :
# endregion
from keras.constraints import Constraint
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Dense
import numpy as np
from keras import initializers
from binary_ops import binarize
from keras.layers import Conv2D


class Clip(Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_cionfig(self):
        return {"min_value": self.min_value,
                "max_value": self.max_value}


class BinaryDense(Dense):
    """
    二值化全链接层
    """

    def __init__(self, uints, H=1, kernel_lr_multiplier='Glorot', bias_lrmultiplier=None, **kwargs):
        super(BinaryDense, self).__init__(uints, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lrmultiplier

    def build(self, input_shape):
        assert len(input_shape) >= 2
        inputs_dim = input_shape[1]
        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (inputs_dim + self.units)))
            print("Glorot H:{}".format(self.H))
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(1. / np.sqrt(1.5 / (inputs_dim + self.units)))
            print("Glorot learning rate multiplier:{}".format(self.kernel_lr_multiplier))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(inputs_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multiplier = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint0)

        else:
            self.lr_multiplier = [self.kernel_lr_multiplier]
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: inputs_dim})
        self.build = True

    def call(self, inputs):
        binary_keral = binarize(self.kernel, H=self.H)
        output = K.dot(inputs, binary_keral)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {"H": self.H,
                  "kernel_lr_myltiplier": self.kernel_lr_multiplier,
                  "bias_lr_multiplier": self.bias_lr_multiplier}
        base_config = super(BinaryDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BinaryConv2D(Conv2D):
    """
    二值化卷积层
    """

    def __init__(self, filters, kernel_lr_multiplier='Glorot',
                 bias_lr_multiplier=None, H=1, **kwargs):
        super(self, BinaryConv2D).__init__(filters, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found '"None"'')

        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size = (input_dim, self.filters)

        base = self.kernel_size[0] * self.kernel_size[1]
        if self.H == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.H = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
            print("Glorot learning rate multiplier:{}".format(self.bias_lr_multiplier))

        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = initializers.RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=initializers,
                                      name="kernel",
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier, self.bias_lr_multiplier]
            self.bias = self.add_weight((self.out_dim,),
                                        initializer=initializers,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None

        # set inpt spec
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.build = True

    def call(self, inputs):
        bunary_kernel = binarize(self.kernel, H=self.H)
        outputs = K.conv2d(
            inputs,
            bunary_kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format
            )

        if self.activation is not None:
            return self.activation(outputs)

    def get_config(self):
        config = {"H": self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  "bias_lr_multiplier": self.bias_lr_multiplier}
        base_config = super(self, BinaryConv2D).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Aliases
BinaryConvolution2D = BinaryConv2D
