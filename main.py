# -*- coding: utf-8 -*-
'''
Create Date: 2023/09/26
Author: @1chooo (Hugo ChunHo Lin)
Version: v0.0.1
'''

from os.path import join
from Preprocessing import load_and_preprocess_data
from Model import build_model
from Plot import plot_model_results

def main() -> None:

    # Read the data and seperate them into test and train part.
    train_data_dir = join(".", "data", "train_image")
    test_data_dir = join(".", "data", "test_image")
    data_train_X, data_train_Y = load_and_preprocess_data(train_data_dir)
    data_test_X, data_test_Y = load_and_preprocess_data(test_data_dir)

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
    score = model.evaluate(
        data_test_X, 
        data_test_Y, 
        verbose=1,
        batch_size=None,
        steps=None,
        workers=1,
        use_multiprocessing=False,
    )

    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])

    model.save("hugo_cnn_handwriting.keras")

    plot_model_results(train_history)

if __name__ == '__main__':
    main()
