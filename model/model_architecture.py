from model.Video_classification_modules import Conv2Plus1D
from model.Video_classification_modules import add_residual_block
from model.Video_classification_modules import ResizeVideo

from keras import layers
import keras

def create_model(HEIGHT,WIDTH,CHANNELS):

    input_shape = (None, HEIGHT, WIDTH, CHANNELS) # creates a tuple 
    input = layers.Input(shape=input_shape) # takes the tuple (10, HEIGHT, WIDTH, 3) as input
    x = input

    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2)(x)

    model = keras.Model(input, x)

    return model