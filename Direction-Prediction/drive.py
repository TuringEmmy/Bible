#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2020/2/25-13:28
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Bible-drive
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import base64
from io import BytesIO

import cv2
import eventlet
import socketio
from PIL import Image, ImageOps
from flask import Flask
import numpy as np
from keras.models import model_from_json

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def send_control(steering_angle, throttle):
    # steer：模拟器的名字
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


@sio.on('telemetry')
def telemetry(sid, data):
    steering_angle = data['steering_angle']
    throttle = data['throttle']
    speed = data['speed']
    imgString = data['image']
    image = Image.open(BytesIO(base64.b64decode((imgString))))
    image_array = np.asarray(image)
    # 色彩空间转换
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

    image_array = image_array[80:140, 0:320]
    # 正规化
    image_array = cv2.resize(image_array, (128, 128)) / 255 - 0.5
    # 将图像从3维增加一个批处理维度
    transformed_image_array = image_array[None, :, :, :]

    # 预测角度
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    # 油门的大小0.1
    throttle = 0.1
    print(steering_angle, throttle)
    # 发送方向盘转动角度和油门给汽车模拟器
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('connect', sid)
    send_control(0, 0)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Remote Driving')
    # parser.add_argument('model', type=str,
    #                     help='Path to model  definition json. Model weight should be on the same path')
    # args = parser.parse_args()
    #
    # with open(args.model, 'r') as jfile:
    #     model = model_from_json(jfile.read())
    #
    # model.compile('adam', 'mse')
    # weights_file = args.model.repalce('json', 'h5')
    # model.load_weights(weights_file)
    #
    # app = socketio.Middleware(sio, app)
    # eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    from keras.models import load_model
    model = load_model('best_model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)