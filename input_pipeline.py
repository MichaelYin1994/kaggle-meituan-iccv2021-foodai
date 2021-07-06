#!/usr/local/bin python
# -*- coding: utf-8 -*-

# Created on 202106121937
# Author:     zhuoyin94 <zhuoyin94@163.com>
# Github:     https://github.com/MichaelYin1994

'''
本模块(input_pipeline.py)构建数据读取与预处理的pipline，并训练神经网络模型。
'''

import os
import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from dingtalk_remote_monitor import RemoteMonitorDingTalk
from models import build_model_resnet50_v2, build_model_resnet101_v2

GLOBAL_RANDOM_SEED = 65535
np.random.seed(GLOBAL_RANDOM_SEED)
tf.random.set_seed(GLOBAL_RANDOM_SEED)

TASK_NAME = 'iccv_meituan_2021'
GPU_ID = 1

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 限制Tensorflow只使用GPU ID编号的GPU
        tf.config.experimental.set_visible_devices(gpus[GPU_ID], 'GPU')

        # 限制Tensorflow不占用所有显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as e:
        print(e)
# ----------------------------------------------------------------------------


def build_efficentnet_model(verbose=False, is_compile=True, **kwargs):
    '''构造基于imagenet预训练的ResNetV2的模型，并返回编译过的模型。'''

    # 解析preprocessing与model的参数
    # ---------------------
    input_shape = kwargs.pop('input_shape', (None, 224, 224))
    n_classes = kwargs.pop('n_classes', 1000)

    model = tf.keras.Sequential()
    # initialize the model with input shape
    model.add(
        tf.keras.applications.EfficientNetB3(
            input_shape=input_shape, 
            include_top=False,
            weights='imagenet',
            drop_connect_rate=0.5,
        )
    )
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
        256,
        activation='relu', 
        bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)
    ))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

    # 编译模型
    # ---------------------
    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(0.005),
            metrics=['acc'])

    return model


def load_preprocess_train_image(image_size=None):
    '''通过闭包实现参数化的训练集的Image loading。'''

    def load_img(path=None):
        image = tf.io.read_file(path)
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, channels=3),
            lambda: tf.image.decode_gif(image)[0])

        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        image = tf.image.resize(image, image_size)
        return image

    return load_img


def load_preprocess_test_image(image_size=None):
    '''通过闭包实现参数化的测试集的Image loading。'''

    def load_img(path=None):
        image = tf.io.read_file(path)
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, channels=3),
            lambda: tf.image.decode_gif(image)[0])

        image = tf.image.resize(image, image_size)
        return image

    return load_img


if __name__ == '__main__':
    # 全局化的参数列表
    # ---------------------
    IMAGE_SIZE = (512, 512)
    BATCH_SIZE = 16
    NUM_EPOCHS = 128
    # NUM_WARMUP_EPOCHS = 6
    # NUM_HOLD_EPOCHS = 5
    EARLY_STOP_ROUNDS = 10
    MODEL_NAME = 'EfficentNetB3_rtx3090'

    CKPT_DIR = './ckpt/'
    CKPT_FOLD_NAME = '{}_GPU_{}_{}'.format(TASK_NAME, GPU_ID, MODEL_NAME)

    IS_TRAIN_FROM_CKPT = False
    IS_SEND_MSG_TO_DINGTALK = True
    IS_DEBUG = False

    if IS_DEBUG:
        TRAIN_PATH = './data/Train_debug/'
        VALID_PATH = './data/Val_debug/'
        TEST_PATH = './data/Test_debug/Public_test_new/'
    else:
        TRAIN_PATH = './data/Train/'
        VALID_PATH = './data/Val/'
        TEST_PATH = './data/Test/Public_test_new/'
    N_CLASSES = len(os.listdir(TRAIN_PATH))

    # 利用tensorflow的preprocessing方法读取数据集
    # ---------------------
    train_file_full_name_list = []
    train_label_list = []
    for dir_name in os.listdir(TRAIN_PATH):
        full_path_name = os.path.join(TRAIN_PATH, dir_name)
        for file_name in os.listdir(full_path_name):
            train_file_full_name_list.append(
                os.path.join(full_path_name, file_name)
            )
            train_label_list.append(int(dir_name))
    train_label_oht_array = np.array(train_label_list)

    val_file_full_name_list = []
    val_label_list = []
    for dir_name in os.listdir(VALID_PATH):
        full_path_name = os.path.join(VALID_PATH, dir_name)
        for file_name in os.listdir(full_path_name):
            val_file_full_name_list.append(
                os.path.join(full_path_name, file_name)
            )
            val_label_list.append(int(dir_name))
    val_label_oht_array = np.array(val_label_list)

    # Encoding labels
    encoder = OneHotEncoder(sparse=False)
    train_label_oht_array = encoder.fit_transform(
        train_label_oht_array.reshape(-1, 1)
    )
    val_label_oht_array = encoder.fit_transform(
        val_label_oht_array.reshape(-1, 1)
    )

    processor_train_image = load_preprocess_train_image(image_size=IMAGE_SIZE)
    processor_valid_image = load_preprocess_test_image(image_size=IMAGE_SIZE)

    # 构造训练集数据
    train_path_ds = tf.data.Dataset.from_tensor_slices(train_file_full_name_list)
    train_img_ds = train_path_ds.map(
        processor_train_image, num_parallel_calls=mp.cpu_count()
    )
    train_label_ds = tf.data.Dataset.from_tensor_slices(train_label_oht_array)

    train_ds = tf.data.Dataset.zip((train_img_ds, train_label_ds))

    # 构造验证集数据
    val_path_ds = tf.data.Dataset.from_tensor_slices(val_file_full_name_list)
    val_img_ds = val_path_ds.map(
        processor_valid_image, num_parallel_calls=mp.cpu_count()
    )
    val_label_ds = tf.data.Dataset.from_tensor_slices(val_label_oht_array)

    val_ds = tf.data.Dataset.zip((val_img_ds, val_label_ds))

    # 性能设定
    train_ds = train_ds.shuffle(buffer_size=int(32 * BATCH_SIZE))
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(2 * BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(2 * BATCH_SIZE)

    # 随机可视化几张图片
    IS_RANDOM_VISUALIZING_PLOTS = False

    if IS_RANDOM_VISUALIZING_PLOTS:
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.title(int(labels[i]))
                plt.axis('off')
        plt.tight_layout()

    # 构造与编译Model，并添加各种callback
    # ---------------------

    # 各种Callbacks
    # ckpt, lr schule, early stop, warm up, remote moniter
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', mode="max",
            verbose=1, patience=EARLY_STOP_ROUNDS,
            restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                CKPT_DIR + CKPT_FOLD_NAME,
                MODEL_NAME + '_epoch_{epoch:02d}_valacc_{val_acc:.3f}.ckpt'),
            monitor='val_acc',
            mode='max',
            save_weights_only=True,
            save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc',
                factor=0.7,
                patience=3,
                min_lr=0.0000003),
        RemoteMonitorDingTalk(
            is_send_msg=IS_SEND_MSG_TO_DINGTALK,
            model_name=CKPT_FOLD_NAME,
            gpu_id=GPU_ID)
    ]

    # 训练模型
    model = build_efficentnet_model(
        n_classes=N_CLASSES,
        input_shape=IMAGE_SIZE + (3,)
    )

    # 如果模型名的ckpt文件夹不存在，创建该文件夹
    if CKPT_FOLD_NAME not in os.listdir(CKPT_DIR):
        os.mkdir(CKPT_DIR + CKPT_FOLD_NAME)

    # 如果指定ckpt weights文件名，则从ckpt位置开始训练
    if IS_TRAIN_FROM_CKPT:
        latest_ckpt = tf.train.latest_checkpoint(CKPT_DIR + CKPT_FOLD_NAME)
        model.load_weights(latest_ckpt)
    else:
        ckpt_file_name_list = os.listdir(CKPT_DIR + CKPT_FOLD_NAME)

        # https://www.geeksforgeeks.org/python-os-remove-method/
        try:
            for file_name in ckpt_file_name_list:
                os.remove(os.path.join(CKPT_DIR + CKPT_FOLD_NAME, file_name))
        except OSError:
            print('File {} can not be deleted !'.format(file_name))

    history = model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # 生成预测结果
    # ---------------------
    test_file_name_list = os.listdir(TEST_PATH)
    test_file_fullname_list = [TEST_PATH + item for item in test_file_name_list]

    test_path_ds = tf.data.Dataset.from_tensor_slices(test_file_fullname_list)
    processor_test_image = load_preprocess_test_image(image_size=IMAGE_SIZE)
    test_ds = test_path_ds.map(
        processor_test_image,
        num_parallel_calls=mp.cpu_count()
    )
    test_ds = test_ds.batch(BATCH_SIZE)
    test_ds = test_ds.prefetch(buffer_size=int(BATCH_SIZE * 2))
    test_pred_proba = model.predict(test_ds)

    test_pred_df = pd.DataFrame(
        test_file_name_list,
        columns=['Id']
    )
    test_pred_df['Predicted'] = np.argmax(test_pred_proba, axis=1)

    sub_file_name = str(len(os.listdir('./submissions'))) + '_sub.csv'
    test_pred_df.to_csv('./submissions/{}'.format(sub_file_name), index=False)
