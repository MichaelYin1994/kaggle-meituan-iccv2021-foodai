#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106112134
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(model.py)构建并编译各种类型的神经网络模型。
'''

from keras import backend
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ----------------------------------------------------------------------------


def res_block_v1(x, n_filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    '''
    基础的残差模块，其中输入默认为channel last模式。

    @Args:
    ----------
    x: {tensor-like}
        输入的tensor，可以是keras的layer。
    filters: {int-like}
        残差模块Bottleneck结构的filters的数量。
    kernel_size: {int-like}
        残差模块Bottleneck结构的kernel的数量。
    stride: {int-like}
        第一层的stride的大小。

    @References:
    ----------
    [1] https://github.com/keras-team/keras/blob/master/keras/applications/resnet.py#L213

    '''


  """A residual block.
  Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.
  Returns:
    Output tensor for the residual block.
  """
  bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

  if conv_shortcut:
    shortcut = layers.Conv2D(
        4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
  else:
    shortcut = x

  x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
  x = layers.Activation('relu', name=name + '_1_relu')(x)

  x = layers.Conv2D(
      filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
  x = layers.Activation('relu', name=name + '_2_relu')(x)

  x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
  x = layers.BatchNormalization(
      axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

  x = layers.Add(name=name + '_add')([shortcut, x])
  x = layers.Activation('relu', name=name + '_out')(x)
  return x