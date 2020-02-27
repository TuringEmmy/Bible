#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2020/2/27-10:57
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Bible-neural_style_transfer
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img
from keras_preprocessing.image import img_to_array

parse = argparse.ArgumentParser(description="Neural style transfert with Keras")
parse.add_argument('base_image_path', metavar='base', type=str,default='data/IMG_7271.png',
                   help='path to the image to transform')
parse.add_argument('style_reference_image_path', metavar='ref', type=str,default='data/Screenshot.png',
                   help='path to the style reference image')
parse.add_argument('result_fix', metavar='res_prefix', type=str,default='data/result_fix.png',
                   help='prefix for the saved results')

parse.add_argument('--iter', type=int, default=10, required=False,
                   help='number of iterations to run')
parse.add_argument('--content_weight', type=float, default=0.025, required=False,
                   help='content weight')
parse.add_argument('--style_weight', type=float, default=1.0, required=False,
                   help='style weight')
parse.add_argument('--tv_weight', type=float, default=1.0, required=False,
                   help='total variation weight')
args = parse.parse_args()
base_image_path = args.base_image_path
print(base_image_path, type(base_image_path))
base_image_path = base_image_path[5:]
style_reference_image_path = args.style_reference_image_path
style_reference_image_path = style_reference_image_path[25:]
result_predix = args.result_fix

# base_image_path='data/IMG_7271.png'
# style_reference_image_path='data/Screenshot.png'
# result_predix='data/result_fix.png'
iterations = args.iter

# these are the weight of different loss components
total_variable_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# dimensions of the generated pictre
img = load_img(base_image_path)
width, height = img.width, img.height
img_nrows = 400
img_ncols = int(width * img_nrows / height)


# util function to open, resize and format picture into appropriate tensor

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x):
    if K.image_data_format() == "channels_first":
        x = x.reshape((3, img_ncols, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))

    # remove zero center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR-->RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path))
style_reference_iamge = K.variable(preprocess_image(style_reference_image_path))

# this will contain our generated image
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder(1, 3, img_nrows, img_ncols)
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_image, style_reference_iamge, combination_image], axis=0)

# build the VGG19 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weight
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
print('Model loaded.')
