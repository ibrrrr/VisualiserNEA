import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
from keras.utils import np_utils

warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = [20, 20]

# helper to show many images at once, for debug purpose
def show_images(images, labels, shape=(3,3)):
    fig, p = plt.subplots(shape[0], shape[1])
    i = 0
    for x in p:
        for ax in x:
            ax.imshow(images[i])
            ax.set_title(labels[i])
            i += 1
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) 
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255
X_test/=255

number_of_classes = 10
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)
