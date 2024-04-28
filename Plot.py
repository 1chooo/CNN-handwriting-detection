# -*- coding: utf-8 -*-
'''
Create Date: 2023/09/26
Author: @1chooo (Hugo ChunHo Lin)
Version: v0.0.1
'''

import matplotlib.pyplot as plt


def plot_model_results(train_history):
    fig, axes = plt.subplots(
        1, 
        2, 
        figsize=(12, 8)
    )

    axes[0].plot(train_history.history["loss"], label="Training Loss")
    axes[0].plot(train_history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(train_history.history["accuracy"], label="Training Accuracy")
    axes[1].plot(train_history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig("./img/results.jpg", dpi=300)

    # plt.show()
