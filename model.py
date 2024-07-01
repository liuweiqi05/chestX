# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import *
from tensorflow.keras.optimizers import Adam, Adamax

SHAPE = (256, 256, 3)


def init_model():
    cap_model = Xception(weights='imagenet', include_top=False, pooling='avg',
                         input_shape=SHAPE)

    cap_model.trainable = False
    model = Sequential()
    model.add(cap_model)
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.45))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adamax(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model
