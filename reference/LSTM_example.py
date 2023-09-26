from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers.recurrent import LSTM
import numpy as np
import os
import matplotlib.pyplot as plt
import deal_with_data as dwd
train_x , train_y = dwd.i_data.data_preprocess_CNN(os.getcwd()+"\\train_image")
test_x, test_y = dwd.i_data.data_preprocess_CNN(os.getcwd() +"\\test_image")

model = Sequential()
model.add(Dense(units = 128, input_shape = (28,28), activation = 'relu'))
model.add(LSTM(units = 32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units = 10, activation = 'softmax'))
model.summary()
model.compile(loss = "categorical_crossentropy",

                optimizer='adam')
history = model.fit(train_x,
                    train_y,
                    epochs=100,
                    validation_split=0.1,
                    batch_size=32)
score = model.evaluate (test_x, test_y)
print(f"score {score}")
# print('test error:', score[0])
# print('test accuracy:', score[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("train_history")
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss', 'val_loss'], loc ='upper left')
plt.show()
