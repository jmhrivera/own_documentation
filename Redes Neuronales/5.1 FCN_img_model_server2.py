
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

def load_train(path):
    
    features_train = np.load(path +'train_features.npy')
    target_train = np.load(path + 'train_target.npy')
    
    n_elements_train, width_train, height_train = features_train.shape
    # test_n_elements, test_width, test_height = features_test.shape

    features_train = (
        features_train.reshape(
            n_elements_train, width_train, height_train,1) /255)

    return features_train, target_train

def create_model(input_shape):
    model = keras.models.Sequential()

    model.add(
        Conv2D(

        )
    )
    model.add(keras.layers.Dense(
        units=64, input_shape=input_shape, activation='relu'
    ))
    model.add(keras.layers.Dense(
        units=32, input_shape=input_shape, activation='relu'
    ))

    model.add(keras.layers.Dense(
        units=32, input_shape=input_shape, activation='relu'
    ))
    model.add(keras.layers.Dense(
        units=10, input_shape=input_shape, activation='softmax'
    ))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )
    return model

def train_model(model, train_data, test_data,
                 batch_size=32, epochs=25, steps_per_epoch=None, validation_steps=None):

    features_train, target_train = train_data
    features_test, target_test = test_data

    model.fit(
        features_train,
        target_train,
        validation_data=(features_test, target_test),
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps= validation_steps,
        verbose=2,
        shuffle=True
    )

    return model



