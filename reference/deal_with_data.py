from PIL import Image
import numpy as np 
import os
from keras.utils import np_utils
class i_data:
    @staticmethod
    def data_preprocess_CNN(datapath):
        '''input 1*28*28version'''
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
    @staticmethod
    def data_preprocess_RNN(datapath):
        '''input 784version'''
        # img_row, img_col = 28, 28
        datapath = datapath
        data_x = np.zeros(784).reshape(784)
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
                img = (np.array(img) / 255).reshape(784)
                data_x = np.vstack((data_x, img))
                pictureCount += 1
        data_x = np.delete(data_x,[0],0)
        data_x = data_x.reshape(pictureCount,784, 1)
        data_y = np_utils.to_categorical(data_y ,num_class)
        return data_x, data_y