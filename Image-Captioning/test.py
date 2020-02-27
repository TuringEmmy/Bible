#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# region Description
# @Origin   ：-i https://pypi.douban.com/simple
# @Date     : 2/12/20-15:38
# @Motto    : Life is Short, I need use Python
# @Author   : turing
# @Email    : yonglonggeng@163.com
# @Contact   : 微信号：csy_lgy 微信公众号：码龙社 CSDN：https://me.csdn.net/sinat_26745777 知乎：https://www.zhihu.com/people/TuringEmmy
# @Project  : Image-Captioning-test
# endregion


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def tokenizer_test():
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer()
    lines = ['this is good', 'that is a cat']
    tokenizer.fit_on_texts(lines)
    results = tokenizer.texts_to_sequences(['cat is good'])
    print(results[0])


def to_categorical_test():
    from keras.utils import to_categorical
    from keras.preprocessing.sequence import pad_sequences
    max_length = 6
    vocab_size = 661
    seq = [2, 660, 6, 229, 3]

    i = 1
    in_seq, out_seq = seq[:i], seq[i]
    # 填充in_seq,使其长度为max_length
    in_seq = pad_sequences([in_seq], max_length)[0]
    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

    print(in_seq)
    print(out_seq)


def title_text(max_length, tokenizer):
    from keras.preprocessing.sequence import pad_sequences
    import numpy as np
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])
        sequence = pad_sequences(sequence, maxlen=max_length)

        output = model.predict([photo_feature, sequence])
        integer = np.argmax(output)


def evaluate_model(model, captions, photo_features, tokenizer, max_length=40):
    """
    计算训练好的神经网络产生的标题的质量，根据4个BLEU分数来评估
    :param model: 训练好的产生标题的神经网络
    :param captions: 测试数据集，key为文件名(不带.jpg   )，value为图像特征
    :param photo_features: dict，key为文件名，value为图像特征
    :param tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
    :param max_length: 训练集中的标题的最大长度
    :return:
    """


def evalue_text():
    from nltk.translate.bleu_score import corpus_bleu
    references = [[['there', 'is', 'a', 'cat', 'and', 'a', 'dog'],
                   ['there','is','']]]
    candidates = [['there', 'is', 'a', 'cat', 'and', 'a', 'pig']]
    score = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))  # 比较两个句子的差异
    print(score)


if __name__ == '__main__':
    # tokenizer_test()
    # to_categorical_test()
    # title_text()
    evalue_text()
