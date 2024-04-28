# -*- coding: utf-8 -*-
'''
Create Date: 2023/09/26
Author: @1chooo (Hugo ChunHo Lin)
Version: v0.0.1
'''

import os

import numpy as np
import tensorflow as tf
from PIL import Image

# Create the function to read the data we have, then preprocess them. 
# We seperate to xData and yData.

def load_and_preprocess_data(
        datapath: str
    ) -> tuple[
        np.ndarray[float], 
        np.ndarray[float],
    ]:
    # Define how big of the image.
    imageRow, imageColumn = 28, 28

    # dataX is the place where we store the image.
    # And we pre-announce the image.
    dataX = np.zeros((28, 28)).reshape(1, 28, 28)
    picture_count = 0

    # dataY is the place where we store the label of image, 
    # and the important thing is that every image have their 
    # own label then stick to it.
    dataY = []

    # We have ten types of data.
    number_class = 10

    for root, dirs, files in os.walk(datapath) :
        for f in files :
            label = int(root.split("/")[4])
            # print(label)
            dataY.append(label)

            # Get the full path of image then we open it.
            full_path = os.path.join(root, f)
            # print(fullPath)
            image = Image.open(full_path)

            # We make the image to the correct size.
            # And we found that the image will be transformed into 2-D array.
            image = (np.array(image) / 255).reshape(1, 28, 28)
            # print(image)  
            dataX = np.vstack((dataX, image))
            picture_count += 1

    # Delete the data we announce at first.
    dataX = np.delete(dataX, [0], 0)

    # Reshape the image again and change them into gray level.
    dataX = dataX.reshape(picture_count, imageRow, imageColumn, 1)

    # Diverse the label into every class we know in line 33 we have announced.
    dataY = tf.keras.utils.to_categorical(dataY, number_class)

    return (
        dataX, 
        dataY
    )
