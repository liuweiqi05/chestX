import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = (256, 256)
Batch = 32


def train_d(tr_path):
    train_data = tr_path
    filepaths = []
    labels = []

    folds = os.listdir(train_data)

    for fold in folds:
        foldpath = os.path.join(train_data, fold)
        filelist = os.listdir(foldpath)
        for pic in filelist:
            fpath = os.path.join(foldpath, pic)
            filepaths.append(fpath)
            labels.append(fold)

    file_path_series = pd.Series(filepaths, name='filepath')
    Label_path_series = pd.Series(labels, name='label')
    df_train = pd.concat([file_path_series, Label_path_series], axis=1)

    return df_train


def test_d(ts_path):
    train_data = ts_path
    filepaths = []
    labels = []

    folds = os.listdir(train_data)

    for fold in folds:
        foldpath = os.path.join(train_data, fold)
        filelist = os.listdir(foldpath)
        for pic in filelist:
            fpath = os.path.join(foldpath, pic)
            filepaths.append(fpath)
            labels.append(fold)

    file_path_series = pd.Series(filepaths, name='filepath')
    Label_path_series = pd.Series(labels, name='label')
    df_test = pd.concat([file_path_series, Label_path_series], axis=1)

    return df_test


def val_d(vl_path):
    train_data = vl_path
    filepaths = []
    labels = []

    folds = os.listdir(train_data)

    for fold in folds:
        foldpath = os.path.join(train_data, fold)
        filelist = os.listdir(foldpath)
        for pic in filelist:
            fpath = os.path.join(foldpath, pic)
            filepaths.append(fpath)
            labels.append(fold)

    file_path_series = pd.Series(filepaths, name='filepath')
    Label_path_series = pd.Series(labels, name='label')
    df_val = pd.concat([file_path_series, Label_path_series], axis=1)

    return df_val


def init_train_ds(tr_path):
    train_data = tr_path
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_data,
        validation_split=0.1,
        subset='training',
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=Batch)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_data,
        validation_split=0.1,
        subset='validation',
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=Batch)

    train_ds, val_ds = train_ds.map(lambda x, y: (x / 255.0, y)), val_ds.map(lambda x, y: (x / 255.0, y))

    return train_ds, val_ds


def init_test_ds(ts_path):
    test_data = ts_path
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data,
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=Batch)

    # test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

    return test_ds
