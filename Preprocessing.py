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

# Create the function to read the data we have, then preprocess them. 
# We seperate to xData and yData.

def dataXYPreprocess (datapath) :
    # Define how big of the image.
    imageRow, imageColumn = 28, 28
    datapath = datapath

    # dataX is the place where we store the image.
    # And we pre-announce the image.
    dataX = np.zeros((28, 28)).reshape(1, 28, 28)
    pictureCount = 0

    # dataY is the place where we store the label of image, 
    # and the important thing is that every image have their 
    # own label then stick to it.
    dataY = []

    # We have ten types of data.
    numberClass = 10

    for root, dirs, files in os.walk(datapath) :
        for f in files :
            label = int(root.split("/")[4])
            # print(label)
            dataY.append(label)

            # Get the full path of image then we open it.
            fullPath = os.path.join(root, f)
            # print(fullPath)
            image = Image.open(fullPath)

            # We make the image to the correct size.
            # And we found that the image will be transformed into 2-D array.
            image = (np.array(image) / 255).reshape(1, 28, 28)
            # print(image)  
            dataX = np.vstack((dataX, image))
            pictureCount += 1

    # Delete the data we announce at first.
    dataX = np.delete(dataX, [0], 0)

    # Reshape the image again and change them into gray level.
    dataX = dataX.reshape(pictureCount, imageRow, imageColumn, 1)

    # Diverse the label into every class we know in line 33 we have announced.
    dataY = np_utils.to_categorical(dataY, numberClass)
    # print(dataY)
    return dataX, dataY