
import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, AveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def load_train(path):
    train_datagen = ImageDataGenerator(
        # validation_split=0.25,
        rescale=1/255.,
        horizontal_flip=True,
        vertical_flip=True,
        # width_shift_range=0.2,
        # height_shift_range=0.2
    )

    train_datagen_flow = train_datagen.flow_from_directory(
        directory=path,
        target_size=(150,150),
        batch_size=16,
        class_mode='sparse',
        # subset='training',
        seed=12345
    )
    
    return train_datagen_flow

def create_model(input_shape=(150,150,3)):
    model= Sequential()

    # model.add(
    #     Input(input_shape=input_shape)
    # )
    model.add(Conv2D(
        filters=6,
        kernel_size=(5,5),
        activation='tanh',
        padding='same',
        input_shape=input_shape
               )
    )

    model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(AveragePooling2D(pool_size=(2,2)))

    model.add(Conv2D(
        filters=16, 
        kernel_size=(5, 5),
        padding='valid',
        activation='tanh'))

    model.add(MaxPool2D(pool_size=(2, 2)))
    # model.add(AveragePooling2D(pool_size=(2,2)))


    # model.add(Conv2D(
    #     filters=120,
    #     kernel_size=(5, 5),
    #     activation='tanh'))

    model.add(Flatten())
    model.add(Dense(64,activation='relu'))
    model.add(Dense(12,activation='softmax'))
    
    optimizer= Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['acc']
    )

    return model


def train_model(model, train_data, test_data, batch_size=None,
             epochs=10, steps_per_epoch=None,
               validation_steps=None):
    
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data,
            validation_data=test_data,
            batch_size=batch_size,
            epochs= epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2
             )
    return model