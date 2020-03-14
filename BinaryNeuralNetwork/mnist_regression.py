#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 3/13/20-18:47
# @Motto    : Life is Short, I need use Python
# @Author   : TuringEmmy
# @Email    : yonglonggeng@163.com
# @Contact   : wechat：csy_lgy  CSDN：https://me.csdn.net/sinat_26745777 
# @Project  : Bible-mnist_regression
# endregion

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 加载数据
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# 定义回归模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 预测值
y = tf.matmul(x, W) + b

# 定义损失函数和优化器
# 输入真实值的占位符
y_ = tf.placeholder(tf.float32, [None, 10])

corss_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# 采用SGD作为优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(corss_entropy)

# 训练模型
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={
        x: batch_xs,
        y: batch_ys
    })

# evalue model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={
    x: mnist.test.images,
    y_: mnist.test.labels
}))
