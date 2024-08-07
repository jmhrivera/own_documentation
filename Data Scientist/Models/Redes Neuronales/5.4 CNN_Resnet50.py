from tensorflow.keras.layers import Conv2D, Flatten, Dense, AvgPool2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
import numpy as np


## Funciona para CNN resnet50 con problemas de clasificación

def load_train(path):
    train_datagen = ImageDataGenerator(rescale=1/255.)

    train_datagen_flow = train_datagen.flow_from_directory(
        path,
        target_size=(150, 150),
        batch_size=16,
        class_mode='sparse',
        seed=12345)

    return train_datagen_flow

def create_model(input_shape=(150, 150, 3)):
    backbone = ResNet50(
        input_shape=(150, 150, 3), # Tamaño de la imagen width, height, channels
        # classess=1000, #Neuronas
        weights='imagenet', # Indica que los pesos se inicializaran de ImageNet
    #Image net es una base de datos de imagenes.
        include_top=False #indica que hay dos capas al final de 
    # ResNet (GlobalAveragePooling2D y Dense). Si la 
    #configuras False, estas dos capas no estarán presentes.

    # backbone.trainable = False # congela ResNet50 sin la parte superior

    # Congelar la red te permite evitar el exceso de entrenamiento e 
    #incrementar la tasa de aprendizaje de la red (el descenso de gradiente no necesitará contar derivadas para las capas congeladas).


    )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(12, activation='softmax')) 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

    return model

def train_model(model, train_data, test_data, batch_size=None, epochs=5, steps_per_epoch=None, validation_steps=None):
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)
    
    return model