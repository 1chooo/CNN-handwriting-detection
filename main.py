# -*- coding: utf-8 -*-
'''
Create Date: 2023/09/26
Author: @1chooo (Hugo ChunHo Lin)
Version: v0.0.1
'''

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

    model.save("hugo_cnn_handwriting.h5")

    plot_model_results(train_history)
