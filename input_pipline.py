#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106121937
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(input_pipline.py)构建数据读取与预处理的pipline，并训练神经网络模型。
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam

GLOBAL_RANDOM_SEED = 192
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# ----------------------------------------------------------------------------


def build_model(verbose=False, is_compile=True, **kwargs):
    '''构造preprocessing与model的pipline，并返回编译过的模型。'''

    # 解析preprocessing与model的参数
    # ---------------------
    input_shape = kwargs.pop('input_shape', (None, 224, 224))
    n_classes = kwargs.pop('n_classes', 1000)

    # 构造data input与preprocessing的pipline
    # ---------------------
    layer_input = keras.Input(shape=input_shape, name='layer_input')

    layer_data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip('horizontal'),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ])
    layer_input_aug = layer_data_augmentation(layer_input)
    layer_input_aug = layers.experimental.preprocessing.Rescaling(
        1 / 255)(layer_input)

    # 构造Model的pipline
    # ---------------------
    x = layers.Conv2D(32, 3, strides=2, padding='same')(layer_input_aug)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x

    for size in [128, 128]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(size, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding='same')(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    if n_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = n_classes

    x = layers.Dropout(0.5)(x)
    layer_output = layers.Dense(units, activation=activation)(x)

    # 编译模型
    # ---------------------
    model = Model(
        layer_input, layer_output)

    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(0.0001),
            metrics=['acc'])

    return model


if __name__ == '__main__':
    # 全局化的参数列表
    # ---------------------
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 4
    NUM_EPOCHS = 1
    EARLY_STOP_ROUNDS = 15
    MODEL_NAME = 'resnet50v2'
    CKPT_PATH = './ckpt/'

    IS_TRAIN_FROM_CKPT = True
    IS_SEND_MSG_TO_DINGTALK = False
    IS_DEBUG = True

    if IS_DEBUG:
        TRAIN_PATH = './data/Train_debug/'
        VALID_PATH = './data/Val_debug/'
        TEST_PATH = './data/Test_debug/Public_test_new/'
    else:
        TRAIN_PATH = './data/Train/'
        VALID_PATH = './data/Val/'
        TEST_PATH = './data/Test/Public_test_new/'

    # 利用tensorflow的preprocessing方法读取数据集
    # ---------------------
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH,
        label_mode='categorical',
        shuffle=True,
        validation_split=0,
        seed=GLOBAL_RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        VALID_PATH,
        label_mode='categorical',
        shuffle=True,
        validation_split=0,
        seed=GLOBAL_RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_PATH,
        shuffle=False,
        label_mode=None,
        validation_split=0,
        seed=GLOBAL_RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE)

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    test_ds = test_ds.prefetch(buffer_size=32)

    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype('uint8'))
    #         plt.title(int(labels[i]))
    #         plt.axis('off')
    # plt.tight_layout()

    # 构造与编译Model，并添加各种callback
    # ---------------------

    # 各种Callbacks
    # ckpt, lr schule, early stop, warm up, remote moniter
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc", mode="max",
            verbose=1, patience=EARLY_STOP_ROUNDS,
            restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                CKPT_PATH,
                MODEL_NAME + '_epoch_{epoch:02d}_valacc_{val_acc:.2f}.ckpt'),
            monitor='val_acc',
            mode='max',
            save_weights_only=True,
            save_best_only=True)]

    # 训练模型
    model = build_model(n_classes=21, input_shape=IMAGE_SIZE + (3,))

    # 如果指定ckpt weights文件名，则从ckpt位置开始训练
    if IS_TRAIN_FROM_CKPT:
        latest_ckpt = tf.train.latest_checkpoint(CKPT_PATH)
        model.load_weights(latest_ckpt)
    else:
        ckpt_file_name_list = os.listdir(CKPT_PATH)

        # https://www.geeksforgeeks.org/python-os-remove-method/
        try:
            for file_name in ckpt_file_name_list:
                os.remove(os.path.join(CKPT_PATH, file_name))
        except OSError:
            print('File {} can not be deleted !'.format(file_name))

    history = model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks)

    # 生成预测结果
    # ---------------------
    test_file_name_list = os.listdir(TEST_PATH)

    test_pred_df = pd.DataFrame(
        test_file_name_list, columns=['Id'])
    test_pred_df['Predicted'] = np.argmax(model.predict(test_ds), axis=1)

    test_pred_df.to_csv('./submissions/sub.csv', index=False)
