import os
import random
import numpy as np
import keras
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.utils import np_utils
from matplotlib import pyplot as plt

def build_model():
    # Construct the "sequential" model.
    model = Sequential()

    # Construct the "Convolutional layer" and 
    # set the output space, kernal size, 
    # and take "relu" as the activation function
    model.add(Conv2D(
        32, 
        kernel_size=(3, 3), 
        activation="relu", 
        input_shape=(28, 28, 1)
    ))

    # Construct "pooling layer" and set the size.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Construct the "Convolutional layer" then this time 
    # we set the size of the fillter to 64 and 
    # the others situation is still the same.
    model.add(Conv2D(64, (3, 3), activation="relu"))

    # Construct "pooling layer" and fetch the maximum of it.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # We dropout in the 25 percent to prevent over fitting.
    model.add(Dropout(0, 1))

    # made mult-D output into 1-D because we want to link to "fully-connected layer"
    model.add(Flatten())

    # Then dropout 10 percent of to prevent over fitting.
    model.add(Dropout(0.1))

    # Contact all layer we have constructed, and the amount of output is 128.
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.25))

    # Classify all results through the activation "softmax", 
    # and the total of the class is 10.
    model.add(Dense(units=10, activation="softmax"))


    # We want to start compile the result; 
    # therefore we pick up the method of "loss", "optimizer", and "Effectiveness"
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam", 
        metrics=["accuracy"]
    )

    return model