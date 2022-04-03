# ------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import sys
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical
import zipfile as zip
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import np_utils

# ------------------------------------------------------------------------------------------------------------------

# root = zip.ZipFile('Animal Image Dataset.zip')
# root.extractall('new')
# root.close()

OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
# DATA_DIR = OR_PATH + os.path.sep + 'animalclassification' + os.path.sep +'new'
DATA_DIR = OR_PATH + os.path.sep + 'new' + os.path.sep
sep = os.path.sep
os.chdir(DATA_DIR)  # Come back to the folder where the code resides , all files will be left on this directory


def load_images(folder):
    images = []
    for i in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, i))
        if img is not None:
            images.append(img)
    return images


animal = os.listdir()
data = []
for i in animal:
    load_images(i)
    data.append(load_images(i))

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 30
BATCH_SIZE = 512
DROPOUT = 0.5
num_classes = 10
img_rows, img_cols = 28, 28
# ------------------------------------------------------------------------------------------------------------------
(X_train, Y_train), (X_test, Y_test) = data
# # building the input vector from the 32x32 pixels
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# building a linear stack of layers with the sequential model
model = Sequential()

# convolutional layer
model.add(Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(10, activation='softmax'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
