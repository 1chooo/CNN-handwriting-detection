# -*- coding: utf-8 -*-
'''
Create Date: 2023/09/20
Author: @1chooo
Version: v0.0.1
'''

# Import the package we need.
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
from Preprocessing import dataXYPreprocess
from Model import build_model
from Plot import plot_model_results

if __name__ == '__main__':

    # Read the data and seperate them into test and train part.
    data_train_X, data_train_Y = dataXYPreprocess("./data/train_image")
    data_test_X, data_test_Y = dataXYPreprocess("./data/test_image")

    model = build_model()

    # training and set the times to 500.
    train_history = model.fit(
        data_train_X, 
        data_train_Y, 
        batch_size=32, 
        epochs=500, 
        verbose=1, 
        validation_split=0.1,
    )

    # Show the result.
    score = model.evaluate(data_test_X, data_test_Y, verbose=0)

    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])

    plot_model_results(train_history)
