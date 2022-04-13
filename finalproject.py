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
from sklearn.model_selection import train_test_split
# ------------------------------------------------------------------------------------------------------------------

# root = zip.ZipFile('Animal Image Dataset.zip')
# root.extractall('Data')
# root.close()

OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
# DATA_DIR = OR_PATH + os.path.sep + 'Deep-Learning/animalclassification/Data'

DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH) # Come back to the folder where the code resides , all files will be left on this directory

random_seed = 42
train_size=0.8

def load_img(dir):
    img_id = []
    for root, animal_folder, img_files in os.walk(dir):
        for i in img_files:
            img_id.append(os.path.join(root, i))
    labels = []
    for j in img_id:
        class_name = j.split('/')[-2]
        if class_name == 'panda':
            labels.append(0)
        elif class_name == 'cow':
            labels.append(1)
        elif class_name == 'spider':
            labels.append(2)
        elif class_name == 'butterfly':
            labels.append(3)
        elif class_name == 'hen':
            labels.append(4)
        elif class_name == 'sheep':
            labels.append(5)
        elif class_name == 'squirrel':
            labels.append(6)
        elif class_name == 'elephant':
            labels.append(7)
        elif class_name == 'monkey':
            labels.append(8)
        elif class_name == 'cats':
            labels.append(9)
        elif class_name == 'horse':
            labels.append(10)
        elif class_name == 'dogs':
            labels.append(11)
    ##shuffle
    data = np.array([img_id, labels])
    data = data.transpose()
    np.random.shuffle(data)
    image_list = list(data[:, 0])
    label_list = list(data[:, 1])
    df = pd.DataFrame((image_list, label_list)).T
    df.rename(columns={0: "img_id", 1: 'target_class'}, inplace=True)
    # Divide the training data into training (80%) and validation (20%)
    df_train, df_test = train_test_split(df, train_size=train_size, random_state=random_seed)
    # Reset the index
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    df_train['split'] = 'train'
    df_test['split'] = 'test'
    print(df_train.head(5), df_train.shape)
    print(50*'-')
    print(df_test.head(5), df_test.shape)
    return df_train, df_test




if __name__ == "__main__":
    dir = DATA_DIR
    # df = pd.DataFrame(load_img(dir)).T
    # df.rename(columns={0: "img_id", 1: 'target_class'}, inplace=True)
    # print(df.head())
    load_img(dir)

# # Directory with training horse pictures
# train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
#
# # Directory with our training human pictures
# train_human_dir = os.path.join('/tmp/horse-or-human/humans')
#
#
# animal = os.listdir()
# data = []
# for i in animal:
#     load_images(i)
#     data.append(load_images(i))


# # %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
# LR = 1e-3
# N_EPOCHS = 30
# BATCH_SIZE = 512
# DROPOUT = 0.5
# num_classes = 10
# img_rows, img_cols = 28, 28
# # ------------------------------------------------------------------------------------------------------------------
# # train_ds, test_ds = tfds.load('mnist', split=['train', 'test[50%]'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# # # building the input vector from the 32x32 pixels
# X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
# X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# # building a linear stack of layers with the sequential model
# model = Sequential()
#
# # convolutional layer
# model.add(Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3)))
#
# # convolutional layer
# model.add(Conv2D(75, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(125, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# # flatten output of conv
# model.add(Flatten())
#
# # hidden layer
# model.add(Dense(500, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(250, activation='relu'))
# model.add(Dropout(0.3))
# # output layer
# model.add(Dense(10, activation='softmax'))
#
# # compiling the sequential model
# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#
# # training the model for 10 epochs
# model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
