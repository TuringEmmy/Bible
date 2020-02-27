#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2020/2/25-07:34
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Direction-Prediction-train
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import math
import pickle

from keras import callbacks
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Flatten, Dense, Dropout
import json
import cv2
from keras.optimizers import SGD
from sklearn.utils import shuffle
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

SEED = 23


def get_model(shape):
    """
    预测方向盘角度：以图像为输入，预测方向盘你够胖的转动角度
    :param shape: 输入图像的尺寸，例如（128， 128，3）
    :return:网络模型
    """

    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=2, input_shape=shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu', input_shape=shape))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu'))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='valid'))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    sgd = SGD()
    model.compile(optimizer=sgd, loss='mean_squared_error')

    # model.summary()

    # save model to json
    with open('autopilot_model.json', 'w') as file:
        file.write(json.dumps(json.loads(model.to_json()), indent=2))
    return model


def image_transformation(img_address, data_dir, degree):
    """
    读取图片
    :param img_address:
    :param data_dir:
    :return:
    """

    img = cv2.imread(data_dir + img_address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img, degree)


def batch_generator(x, y, batch_size, shape, training=True, data_dir='data/', monitor=True, yiedXY=True,
                    discart_rate=0.95):
    """
    产生批处理的数据generator
    :param x:文件的list
    :param y:方向盘的角度
    :param batch_size:批处理大小
    :param shape:
    :param training:值为True时产生训练数据；值为false时产很validation数据
    :param data_dir:数据目录，包含一个IMG文件夹
    :param monitor:保存一个batch的样本'X_batch_sample.npy'和'y_bag.npy'
    :param yiedXY:为True时，反悔（X，Y）;为false时，反返回X only
    :param discart_rate:随机丢弃部分为零的数据

    :return:
    """
    if training:
        y_bag = []
        x, y = shuffle(x, y)
        new_x = x
        new_y = y
    else:
        new_x = x
        new_y = y

    offset = 0
    while True:
        X = np.empty((batch_size, *shape))
        Y = np.empty((batch_size, 1))

        for example in range(batch_size):
            img_address, img_steering = new_x[example + offset], new_y[example + offset]
            if training:
                img, img_steering = image_transformation(img_address, data_dir, img_steering)
            else:
                img, _ = image_transformation(img_address, data_dir, 1)

            # 正则化，并归一化
            X[example, :, :, :] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1])) / 255 - 0.5
            Y[example] = img_steering

            if training:
                y_bag.append(img_steering)

            '''
            到达原来数据的结尾，从新开始
            '''
            if (example + 1) + offset > len(new_y) - 1:  # 需要继续理解
                x, y = shuffle(x, y)
                new_x = x
                new_y = y
                offset = 0

        if yiedXY:
            yield (X, Y)
        else:
            yield X

        offset += batch_size
        if training:
            np.save('y_bag.npy', np.array(y_bag))
            np.save('Xbatch_sample.npy', X)


def horizontal_flip(img, degree):
    """
    按照50%的概率水平翻转
    :param img: 输入图像
    :param degree: 输入图像反翻转的角度
    :return:
    """
    choice = np.choose([0, 1])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree
    return (img, degree)


def random_birghtness(img, degree):
    """
    随机调整输入图像的亮度，调整强度于（0.1,1）之间
    :param img: 输入图像
    :param degree: 输入图像对于转动角度
    :return:
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # 调整亮度VL:alpha*V
    alpha = np.random.uniform(low=0.1, high=1.0, size=None)
    v = hsv[:, :, 2]
    v = v * alpha
    hsv[:, :, 2] = v.astype('uint8')
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)

    return (rgb, degree)


def left_right_random_swap(img_address, degree, degree_corr=1.0 / 4):
    """
    随机从左中右图像中选择一张图像，并响应的调整转动的角度
    :param img_address: 中间图像的的文件路径
    :param degree: 中间图像对于方向盘转动角度
    :param degree_corr: 方向盘转动角度调整的值
    :return:
    """
    swap = np.random.choice(['L', 'R', 'C'])

    if swap == 'L':
        img_address = img_address.replace('center', 'left')
        corrected_label = np.arctan(math.tan(degree) + degree_corr)
    elif swap == 'R':
        img_address = img_address.replace('center', 'right')
        corrected_label = np.arctan(math.tan(degree) - degree_corr)
    else:
        corrected_label = degree_corr
    return (img_address, corrected_label)


def discard_zero_steering(degrees, rate):
    """
    从角度为零的index中随机选择部分index返回
    :param degrees: 输入的角度值
    :param rate: 丢弃率，如果rate=0.8，意味着80%的index会被返回，用于丢弃
    :return:
    """
    steering_zero_idx = np.where(degrees=0)
    steering_zero_idx = steering_zero_idx[0]
    size_dell = int(len(steering_zero_idx) * rate)

    return np.random.choice(steering_zero_idx, size=size_dell, replace=False)


if __name__ == '__main__':
    data_path = "data/"
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        log = []
        for row in csv_reader:
            log.append(row)

    log = np.array(log)
    # log = log[1:, :]

    # 判断图像文件数量是否等于csv日志文件中的数量
    ls_imgs = glob.glob(data_path + 'IMG/*.jpg')
    assert len(ls_imgs) == len(log) * 3, '日志数和图片数量不匹配'

    # 使用20%的数据作为测试数据
    validation_ration = 0.2
    shape = (128, 128, 3)
    batch_size = 32
    nb_epoch = 2

    x_ = log[:, 0]
    y_ = log[:, 3].astype(float)
    x_, y_ = shuffle(x_, y_)
    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validation_ration, random_state=SEED)

    print('batch size: {}'.format(batch_size))
    print('train set size: {} | validation size: {}'.format(len(X_train), len(X_val)))

    samples_per_epoch = batch_size

    # 使得validation数据量的大小为batch_size的整数倍
    nb_val_samples = len(y_val) - len(y_val) % batch_size

    model = get_model(shape)
    print(model.summary())

    # 根据validation loss保存最优模型
    save_best = callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                          mode='min')

    # 如果训练持续没有validation loss提升，提前结束训练
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
    callbacks_list = [early_stop, save_best]

    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, shape, training=True),
                                  steps_per_epoch=samples_per_epoch, validation_steps=nb_val_samples // batch_size,
                                  validation_data=batch_generator(X_val, y_val, batch_size, shape, training=False,
                                                                  monitor=False), epochs=nb_epoch, verbose=1,
                                  callbacks=callbacks_list)

    # graph
    tbCallBack = callbacks.TensorBoard(log_dir='./Graph', write_grads=True)

    with open('./TrainHistory_dict.p', 'wb') as pi:
        pickle.dump(history.history, pi)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train VS validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('train_val_los.jpg')

    # 保存模型
    with open('mode.json', 'w') as f:
        f.write(model.to_json())
    model.save('model.h5')
    print('Done')
