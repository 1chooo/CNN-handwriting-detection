# -*- coding: utf-8 -*-
'''
Create Date: 2023/09/26
Author: @1chooo
Version: v0.0.1
'''

import matplotlib.pyplot as plt

def plot_model_results(train_history):
    # Make the result visualize.
    plt.plot(train_history.history["loss"])
    plt.plot(train_history.history["val_loss"])

    plt.title("Train History")
    plt.ylabel("loss")
    plt.xlabel("Epoch")

    plt.legend(["loss", "val_loss"], loc="upper left")
    plt.savefig("./src/img/result.jpg", dpi=300)

    plt.show()