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

GPU_ID = 0

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


def build_model(verbose=False, is_compile=True, **kwargs):
    '''构造preprocessing与model的pipline，并返回编译过的模型。'''
    network_type = kwargs.pop('network_type', 'resnet50')

    # 解析preprocessing与model的参数
    # ---------------------
    input_shape = kwargs.pop('input_shape', (None, 224, 224))
    n_classes = kwargs.pop('n_classes', 1000)

    # 构造data input与preprocessing的pipline
    # ---------------------
    layer_input = keras.Input(shape=input_shape, name='layer_input')

    # 构造Model的pipline
    # ---------------------
    if 'resnet50' in network_type: 
        x = build_model_resnet50_v2(layer_input)
    elif 'resnet101' in network_type:
        x = build_model_resnet101_v2(layer_input)

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
    model = Model(layer_input, layer_output)

    if verbose:
        model.summary()

    if is_compile:
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(0.0001),
            metrics=['acc'])

    return model


def load_preprocess_image(image_size=None):
    '''通过闭包实现参数化的Image loading。'''

    def fcn(path=None):
        image = tf.io.read_file(path)
        image = tf.cond(
            tf.image.is_jpeg(image),
            lambda: tf.image.decode_jpeg(image, channels=3),
            lambda: tf.image.decode_gif(image)[0])
        image = tf.image.resize(image, image_size)
        image = layers.experimental.preprocessing.Rescaling(1. / 255.)(image)

        return image
    return fcn


if __name__ == '__main__':
    # 全局化的参数列表
    # ---------------------
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 64
    NUM_EPOCHS = 128
    EARLY_STOP_ROUNDS = 7
    MODEL_NAME = 'resnet50v2_iccv2021_rtx3090'
    CKPT_PATH = './ckpt/{}/'.format(MODEL_NAME)

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

    # Construct training dataset
    load_preprocess_train_image = load_preprocess_image(image_size=IMAGE_SIZE)

    train_path_ds = tf.data.Dataset.from_tensor_slices(train_file_full_name_list)
    train_img_ds = train_path_ds.map(
        load_preprocess_train_image, num_parallel_calls=mp.cpu_count()
    )
    train_label_ds = tf.data.Dataset.from_tensor_slices(train_label_oht_array)

    train_ds = tf.data.Dataset.zip((train_img_ds, train_label_ds))

    # Construct validation dataset
    val_path_ds = tf.data.Dataset.from_tensor_slices(val_file_full_name_list)
    val_img_ds = val_path_ds.map(
        load_preprocess_train_image, num_parallel_calls=mp.cpu_count()
    )
    val_label_ds = tf.data.Dataset.from_tensor_slices(val_label_oht_array)

    val_ds = tf.data.Dataset.zip((val_img_ds, val_label_ds))

    # Performance
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
                CKPT_PATH,
                MODEL_NAME + '_epoch_{epoch:02d}_valacc_{val_acc:.3f}.ckpt'),
            monitor='val_acc',
            mode='max',
            save_weights_only=True,
            save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_acc',
                factor=0.7,
                patience=2,
                min_lr=0.0000003),
        RemoteMonitorDingTalk(
            is_send_msg=IS_SEND_MSG_TO_DINGTALK,
            model_name=MODEL_NAME,
            gpu_id=GPU_ID)
    ]

    # 训练模型
    model = build_model(
        n_classes=N_CLASSES,
        input_shape=IMAGE_SIZE + (3,),
        network_type=MODEL_NAME
    )

    # 如果模型名的ckpt文件夹不存在，创建该文件夹
    if MODEL_NAME not in os.listdir('./ckpt'):
        os.mkdir('./ckpt/' + MODEL_NAME)

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
        callbacks=callbacks
    )

    # 生成预测结果
    # ---------------------
    test_file_name_list = os.listdir(TEST_PATH)
    test_file_fullname_list = [TEST_PATH + item for item in test_file_name_list]

    test_path_ds = tf.data.Dataset.from_tensor_slices(test_file_fullname_list)
    load_preprocess_test_image = load_preprocess_image(image_size=IMAGE_SIZE)
    test_ds = test_path_ds.map(
        load_preprocess_test_image,
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
    test_pred_df.to_csv('./submissions/sub.csv', index=False)
