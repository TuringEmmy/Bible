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
import time

import numpy as np
from keras import backend as K
from keras.applications import vgg19
from keras.preprocessing.image import load_img, save_img
from keras_preprocessing.image import img_to_array
from scipy.optimize import fmin_l_bfgs_b

parse = argparse.ArgumentParser(description="Neural style transfert with Keras")
parse.add_argument('base_image_path', metavar='base', type=str, default='data/IMG_7271.png',
                   help='path to the image to transform')
parse.add_argument('style_reference_image_path', metavar='ref', type=str, default='data/Screenshot.png',
                   help='path to the style reference image')
parse.add_argument('result_fix', metavar='res_prefix', type=str, default='data/result_fix.png',
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

# get the symbolic outputs of each "key" layer(we gave them unique names)
outputs_dict = dict([(layer.nmae, layer.output) for layer in model.layers])


# compute the neural style loss
# first we need  to define 4 util function

# the gram matrix of an image tensor (feature-wise outter product)

def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == "channel_first":
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))

    gram = K.dot(features, K.transpose(features))
    return gram


# the style loss is designed to maintain
# the style of the reference image in the generagted image
# it is based on the gram matrices (which capture style) of and  from the generated image

def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4 * (channels ** 2) * (size ** 2))


# an auxiliary loss function
# designed to maintain the content of the base image in the generated image


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


# the 3rd loss function, total variation loss, designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim() == 4
    if K.image_data_format() == "channels_first":
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_ncols - 1, 1:, :])
    return K.sum(K.pow(a + b), 1.15)


# comnina these loss function into a single scalar
loss = K.variable(0.0)
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0:, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features, combination_features)

feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'blcok5_conv1']

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

loss += total_variable_weight + total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)
outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')

    return loss_value, grad_values


# this evaluator class makes it possible to compute loss and gradient in one pass
# while retrieving them via two separate function, "loss"and "grads". This is done because scity optimize requests
# separate functions for loss and gradients, but computing them separately would be inefficient
class Evaluartor(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss
        self.grads_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values


evaluator = Evaluartor()

# run scipy based optimization(L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = preprocess_image(base_image_path)
for i in range(iterations):
    print("start of iteration", i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print("current loss value:", min_val)
    # save current generated image

    img = deprocess_image(x.copy())
    fname = result_predix + '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print("image saved as", fname)
    print('iteration %d completed in %s' % (i, end_time - start_time))
