#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106211120
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(test_models.py)对models.py中的模块进行简单单元测试。
'''

import sys

sys.path.append('..')

import tensorflow as tf
from models import (residual_block_v1, residual_block_v2, residual_module_v1,
                    residual_module_v2)
from tensorflow import keras
from tensorflow.keras import backend, layers

# ----------------------------------------------------------------------------


def test_residual_block_v1():
    '''对于models.residual_block_v1的测试'''
    input_shape = (224, 224, 3)

    # 输入为channel last的一张image
    layer_input = keras.Input(shape=input_shape)

    x = residual_block_v1(
        layer_input, n_filters=128, kernel_size=3, stride=1,
        conv_shortcut=True, name='initial')
    assert [None, 224, 224, 512] == x.get_shape().as_list()

    x = residual_block_v1(
        layer_input, n_filters=128, kernel_size=3, stride=2,
        conv_shortcut=True, name='initial')
    assert [None, 112, 112, 512] == x.get_shape().as_list()

    # 无法通过测试，conv_shortcut短接时为clean的通道
    try:
        x = residual_block_v1(
            layer_input, n_filters=128, kernel_size=3, stride=2,
            conv_shortcut=False, name='initial')
    except ValueError:
        pass

    return None


def test_residual_module_v1():
    '''对于models.residual_module_v1的测试'''
    input_shape = (224, 224, 3)

    # 输入为channel last的一张image
    layer_input = keras.Input(shape=input_shape)

    x = residual_module_v1(
        layer_input, n_filters=64, n_blocks=6, stride=1, name='resnetv1')

    return None


def test_residual_block_v2():
    '''对于models.residual_block_v1的测试'''
    input_shape = (224, 224, 128 * 4)

    # 输入为channel last的一张image
    layer_input = keras.Input(shape=input_shape)

    x = residual_block_v2(
        layer_input, n_filters=128, kernel_size=3, stride=1,
        conv_shortcut=False, name='initial')
    assert [None, 224, 224, 512] == x.get_shape().as_list()

    # 是否使用conv_shortcut都可以通过测试，v2的block的shortcut有特殊
    # feature map缩减的方法
    x = residual_block_v2(
        layer_input, n_filters=128, kernel_size=3, stride=2,
        conv_shortcut=True, name='initial')
    assert [None, 112, 112, 512] == x.get_shape().as_list()

    x = residual_block_v2(
        layer_input, n_filters=128, kernel_size=3, stride=2,
        conv_shortcut=False, name='initial')
    assert [None, 112, 112, 512] == x.get_shape().as_list()

    return None


def test_residual_module_v2():
    '''对于models.residual_block_v1的测试'''
    input_shape = (224, 224, 3)

    # 输入为channel last的一张image
    layer_input = keras.Input(shape=input_shape)

    x = residual_module_v2(
        layer_input, n_filters=256, n_blocks=6, stride=1, name='resnetv2')

    return None


if __name__ == '__main__':
    test_residual_block_v1()
    test_residual_module_v1()

    test_residual_block_v2()
    test_residual_module_v2()
