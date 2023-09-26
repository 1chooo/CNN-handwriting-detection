from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import os
import matplotlib.pyplot as plt
import deal_with_data as dwd
# 建立模型
model = Sequential()

# 此為建立兩層hidden layer的DNN 
model.add(Dense(units = 512, input_dim = 784, activation = 'relu'))

model.add(Dense(units = 256, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(units = 256, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(units = 128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

#查看模型摘要
# model.summary()

# 模型設定訓練
model.compile(loss = 'categorical_crossentropy',
                optimizer='adam',
                metrics = ['accuracy'])

train_x , train_y = dwd.i_data.data_preprocess_RNN(os.getcwd()+"\\train_image")

test_x, test_y = dwd.i_data.data_preprocess_RNN(os.getcwd() +"\\test_image")

history = model.fit(train_x,
                    train_y,
                    epochs=300,
                    validation_split=0.1,
                    batch_size=32)
score = model.evaluate (test_x, test_y)


# print('test error:', score[0])
# print('test accuracy:', score[1])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("train_history")
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss', 'val_loss'], loc ='upper left')
plt.show()