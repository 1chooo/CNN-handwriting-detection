# CNN-based Handwriting Detection and Classification

[![project badge](https://img.shields.io/badge/1chooo-CNN__Handwriting__Dection-informational)](https://github.com/1chooo/CNN-handwriting-dection)
[![conda version](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://docs.conda.io/en/latest/#)
[![conda version](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![conda version](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Made with Python](https://img.shields.io/badge/Python-=3.7-blue?logo=python&logoColor=white)](https://python.org "Go to Python homepage")
[![License](https://img.shields.io/badge/License-MIT-blue)](./LICENSE "Go to license section")

## A brief summary of the project
Author : ChunHo,Lin (1chooo)  

Created time: 2022/06/22  

It is the final project of the course: CE3005-B in NCU which name of the course is "Algorithmics". 

The main goal of this project is to detect the hand writing numbers with the deep learning, **CNN**.

---

## Create Enviroment

conda --version: 4.12.0
 
```
$ conda create --name algML python=3.7
$ conda install tensorflow=1.15.0
$ conda install keras=2.3.1
$ conda install matplotlib
```

You guys can also check more details about the virtual environment in the `requirements.txt` and `env.yml`

## 中文手寫辨識準確率及損失率

![plot](src/img/loss_and_accuracy.png)


## Processing

![plot](src/img/process.png)

## Result

![result](src/img/result.jpg)