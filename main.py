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
            label = int(root.split("/")[5])
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

# Read the data and seperate them into test and train part.
data_train_X, data_train_Y = dataXYPreprocess("./src/handwrite_detect/train_image")
data_test_X, data_test_Y = dataXYPreprocess("./src/handwrite_detect/test_image")


def build_model():
    # Construct the "sequential" model.
    model = Sequential()

    # Construct the "Convolutional layer" and 
    # set the output space, kernal size, 
    # and take "relu" as the activation function
    model.add(Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = (28, 28, 1)))

    # Construct "pooling layer" and set the size.
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # Construct the "Convolutional layer" then this time 
    # we set the size of the fillter to 64 and 
    # the others situation is still the same.
    model.add(Conv2D(64, (3, 3), activation = "relu"))

    # Construct "pooling layer" and fetch the maximum of it.
    model.add(MaxPooling2D(pool_size = (2, 2)))

    # We dropout in the 25 percent to prevent over fitting.
    model.add(Dropout(0, 1))

    # made mult-D output into 1-D because we want to link to "fully-connected layer"
    model.add(Flatten())

    # Then dropout 10 percent of to prevent over fitting.
    model.add(Dropout(0.1))

    # Contact all layer we have constructed, and the amount of output is 128.
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.25))

    # Classify all results through the activation "softmax", 
    # and the total of the class is 10.
    model.add(Dense(units = 10, activation = "softmax"))


    # We want to start compile the result; 
    # therefore we pick up the method of "loss", "optimizer", and "Effectiveness"
    model.compile(loss = "categorical_crossentropy",
                optimizer = "adam", 
                metrics = ["accuracy"])

    return model


model = build_model()

# training and set the times to 500.
trainHistory = model.fit(data_train_X, data_train_Y, 
                         batch_size = 32, 
                         epochs = 500, verbose = 1, 
                         validation_split = 0.1)


# Show the result.
score = model.evaluate(data_test_X, data_test_Y, verbose = 0)

print("Test loss: ", score[0])
print("Test accuracy: ", score[1])


# Make the result visualize.
plt.plot(trainHistory.history["loss"])
plt.plot(trainHistory.history["val_loss"])

plt.title("Train History")
plt.ylabel("loss")
plt.xlabel("Epoch")

plt.legend(["loss", "val_loss"], loc = "upper left")
plt.savefig("./src/img/result.jpg", dpi=300)

plt.show()