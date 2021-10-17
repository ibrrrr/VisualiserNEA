import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

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
train = 


train_image = np.array(train.drop(['label'], axis = 1), dtype = "float32" / 255)
train_image = train_image.reshape(-1, 28, 28, 1)
test = np.array(test, dtype="float32") / 255
test = test.reshape(-1, 28, 28, 1)

show_images(train_image[:25], train_label[:25], shape=(5,5))

train_label = tf.keras.utils.to_categorical(train['label'])


(image_train_mnist, label_train_mnist), (image_test_mnist, label_test_mnist) = mnist.load_data()
image_mnist = np.concatenate((image_train_mnist, image_test_mnist))
label_mnist = np.concatenate((label_train_mnist, label_test_mnist))
image_mnist = image_mnist.reshape(-1,28,28,1)
image_mnist = image_mnist.astype(np.float32) / 255
label_mnist = tf.keras.utils.to_categorical(label_mnist,num_classes=10)
images = np.concatenate((train_image, image_mnist))
labels = np.concatenate((train_label, label_mnist))

# final dataset shape
print("training image dataset shape:", images.shape)
print("training label dataset shape:", labels.shape)

show_images(images[:25], labels[:25], shape=(5,5))
