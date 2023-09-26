from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import os
import matplotlib.pyplot as plt
# 以上為輸入資料集
def data_preprocess(datapath):
    img_row, img_col = 28, 28
    datapath = datapath
    data_x = np.zeros((28,28)).reshape(1,28,28)
    pictureCount = 0
    data_y = []
    num_class = 10
    # 讀取資料夾內所有的資料
    for root, dirs, files in os.walk(datapath):
        for f in files:
            label = int(root.split("\\")[-1])
            data_y.append(label)
            fullpath = os.path.join(root, f)
            img = Image.open(fullpath)
            img = (np.array(img) / 255).reshape(1, 28, 28)
            data_x = np.vstack((data_x, img))
            pictureCount += 1
    data_x = np.delete(data_x,[0],0)
    data_x = data_x.reshape(pictureCount, img_row, img_col, 1)
    data_y = np_utils.to_categorical(data_y ,num_class)
    return data_x, data_y

# 以下為建立模型
# 建立簡單線性模型
model = Sequential()

# 為模型加入捲基層
# input_shape只有第一層捲基層要加入
model.add(Conv2D(filters = 32, kernel_size=(3,3),
            padding = "same", input_shape = (28, 28, 1),
            activation="relu"))

# 加入pooling
model.add(MaxPooling2D(pool_size = (2,2)))

# 加入捲機層跟pooling可以數次(目前只加入一次)

# 第二次加入convolution不需要inputsize
model.add(Conv2D(filters = 64, kernel_size=(3,3),
            padding = "same", activation="relu"))

model.add(MaxPooling2D(pool_size = (2,2)))

# 加入Dropout以防止過度擬合
model.add(Dropout(0.1))

# 加入flatten
model.add(Flatten())

model.add(Dropout(0.1))
# 加入dense(full connected layer)
# 分類問題通常使用softmax當作output的activation function
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(units= 10, activation = 'softmax'))

# 建立模型完成之後進行compile
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

datapath = os.getcwd()

data_train_x, data_train_y = data_preprocess(datapath + "\\" + "train_image")
data_test_x , data_test_y = data_preprocess(datapath + "\\" + "test_image")
# 使用資料進行訓練
train_history = model.fit(data_train_x, data_train_y, 
                        batch_size=32, epochs = 150, 
                        verbose = 1, validation_split=0.1)

# 顯示損失函數和訓練成果
score = model.evaluate(data_test_x, data_test_y, verbose = 0)
print("Test loss", score[0])
print("test accuracy", score[1])

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title("train_history")
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss', 'val_loss'], loc ='upper left')
plt.show()
