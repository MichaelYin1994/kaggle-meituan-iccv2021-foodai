#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106112134
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(models.py)构建并编译各种类型的神经网络模型。此模块大部分代码来自keras application的
部分模块[1]，但是在细节上做了适应。

@References:
----------
[1] https://github.com/keras-team/keras/blob/master/keras/applications
'''

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------------------------------------------------

def residual_block_v1(
        x, n_filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    '''
    一个基础的残差模块，其中输入默认为channel last模式。

    @Args:
    ----------
    x: {tensor-like}
        输入的tensor，可以是keras的layer。
    n_filters: {int-like}
        残差模块Bottleneck结构的filters的数量。
    kernel_size: {int-like}
        残差模块Bottleneck结构的kernel的数量。
    stride: {int-like}
        第一层的stride的大小。
    conv_shortcut: {bool-like}
        是否使用1 * 1的conv层作为short-cut用于升维，对应于论文[2]中的conv通道连接方式。
    name: {str-like}
        block的每一层layer的name prefix。

    @Returns:
    ----------
    构造好的一个残差模块。

    @References:
    ----------
    [1] https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py#L213
    [2] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
    '''
    if stride != 1 and conv_shortcut == False:
        raise ValueError('Input shape mismatch with the shortcut shape !')

    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * n_filters, 1, strides=stride, name=name + '_shortcut_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_shortcut_0_bn')(shortcut)
    else:
        shortcut = x

    # 降维
    x = layers.Conv2D(n_filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    # 特征抽取
    x = layers.Conv2D(
        n_filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    # 升维 + 残差连接
    x = layers.Conv2D(4 * n_filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)

    return x


def residual_block_v2(
        x, n_filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    '''
    一个基础的残差模块，其中输入默认为channel last模式。

    @Args:
    ----------
    x: {tensor-like}
        输入的tensor，可以是keras的layer。
    n_filters: {int-like}
        残差模块Bottleneck结构的filters的数量。
    kernel_size: {int-like}
        残差模块Bottleneck结构的kernel的数量。
    stride: {int-like}
        第一层的stride的大小。
    conv_shortcut: {bool-like}
        是否使用1 * 1的conv层作为short-cut用于升维，对应于论文[2]中的conv通道连接方式。
    name: {str-like}
        block的每一层layer的name prefix。

    @Returns:
    ----------
    构造好的一个残差模块。

    @References:
    ----------
    [1] https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py#L213
    [2] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
    '''
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    preact = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * n_filters, 1, strides=stride, name=name + '_shortcut_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x


def residual_block_v3(
        x, n_filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    pass


def residual_module_v1(x, n_filters, n_blocks, stride=2, name=None):
    '''
    一个基础的残差组件，由一系列的残差模块组成，其中输入默认为channel last模式。

    @Args:
    ----------
    x: {tensor-like}
        输入的tensor，可以是keras的layer。
    n_filters: {int-like}
        残差组件Bottleneck结构的filters的数量。
    n_blocks: {int-like}
        残差组件的block的数量。
    stride: {int-like}
        残差组件的第一层的stride的大小。
    name: {str-like}
        block的每一层layer的name prefix。

    @Returns:
    ----------
    一个构造好的残差组件。
    '''
    # 基础特征图抽取
    x = residual_block_v1(x, n_filters, stride=stride, name=name + '_block1')

    # Residual block叠加，残差连接没有升维操作
    for i in range(2, n_blocks + 1):
        x = residual_block_v1(
            x, n_filters, stride=1, conv_shortcut=False,
            name=name + '_block' + str(i))

    return x


def residual_module_v2(x, n_filters, n_blocks, stride=2, name=None):
    pass


def residual_module_v3(x, n_filters, n_blocks, stride=2, name=None):
    pass

