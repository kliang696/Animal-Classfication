# ------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
import sys
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.python.keras.applications.vgg19 import VGG19
# ------------------------------------------------------------------------------------------------------------------
## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE

# root = zip.ZipFile('Animal Image Dataset.zip')
# root.extractall('Data')
# root.close()
DROPOUT =0.5
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
# DATA_DIR = OR_PATH + os.path.sep + 'Deep-Learning/animalclassification/Data'

DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)  # Come back to the folder where the code resides , all files will be left on this directory

random_seed = 42
train_size = 0.8

n_epoch = 20
BATCH_SIZE = 128

## Image processing
CHANNELS = 3
IMAGE_SIZE = 100

# ------------------------------------------------------------------------------------------------------------------
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, CHANNELS)  # output (n_examples, 26, 26, 16)
        self.convnorm1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(2)  # output (n_examples, 13, 13, 16)
        self.conv2 = tf.keras.layers.Conv2D(32, CHANNELS)  # output (n_examples, 11, 11, 32)
        self.convnorm2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.AveragePooling2D(2)  # output (n_examples, 5, 5, 32)
        self.flatten = tf.keras.layers.Flatten()  # input will be flattened to (n_examples, 32 * 5 * 5)
        self.linear1 = tf.keras.layers.Dense(100)
        self.linear1_bn = tf.keras.layers.BatchNormalization()
        self.linear2 = tf.keras.layers.Dense(OUTPUTS_a)
        self.act = tf.nn.relu
        self.drop = DROPOUT
        self.training = True

    def call(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x)), training=self.training))
        x = self.flatten(self.pool2(self.convnorm2(self.act(self.conv2(x)), training=self.training)))
        x = tf.nn.dropout(self.linear1_bn(self.act(self.linear1(x)), training=self.training), self.drop)
        return self.linear2(x)


def read_train(DATA_DIR):
  train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=random_seed,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE)
  return train_ds

def read_test(DATA_DIR):
    val_ds = tf.keras.utils.image_dataset_from_directory(
      DATA_DIR,
      validation_split=0.2,
      subset="validation",
      seed=random_seed,
      image_size=(IMAGE_SIZE, IMAGE_SIZE),
      batch_size=BATCH_SIZE)
    return val_ds

# ------------------------------------------------------------------------------------------------------------------

def show_data(test_ds):
    img_path = []
    img_id = []
    for i in test_ds.file_paths:
        img_path.append(i)
    for i in img_path:
        a = i.split('/')[-2:]
        img_id.append(a[0] + '/' + a[1])
    labels = []
    for j in img_path:
        class_name = j.split('/')[-2]
        if class_name == 'panda':
            labels.append('panda')
        elif class_name == 'cow':
            labels.append('cow')
        elif class_name == 'spider':
            labels.append('spider')
        elif class_name == 'butterfly':
            labels.append('butterfly')
        elif class_name == 'hen':
            labels.append('hen')
        elif class_name == 'sheep':
            labels.append('sheep')
        elif class_name == 'squirrel':
            labels.append('squirrel')
        elif class_name == 'elephant':
            labels.append('elephant')
        elif class_name == 'monkey':
            labels.append('monkey')
        elif class_name == 'cats':
            labels.append('cats')
        elif class_name == 'horse':
            labels.append('horse')
        elif class_name == 'dogs':
            labels.append('dogs')
    data = np.array([img_id, labels])
    data = data.transpose()
    image_list = list(data[:, 0])
    label_list = list(data[:, 1])
    df = pd.DataFrame((image_list, label_list)).T
    df.rename(columns={0: "id", 1: 'target'}, inplace=True)
    return df


# ------------------------------------------------------------------------------------------------------------------

def save_model(model):
    '''
         receives the model and print the summary into a .txt file
    '''
    with open('summary_{}.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

# ------------------------------------------------------------------------------------------------------------------

def model_definition(models):
    # model = tf.keras.models.Sequential([
    #     # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    #     # This is the first convolution
    #     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     # The second convolution
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     # The third convolution
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     # The fourth convolution
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     # The fifth convolution
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D(2, 2),
    #     # Flatten the results to feed into a DNN
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(OUTPUTS_a, activation='relu')])

    # vgg = VGG19(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    #             classes=OUTPUTS_a)
    # for layer in vgg.layers:
    #     layer.trainable = False
    # model = Sequential()
    # model.add(vgg)
    # model.add(Flatten())
    # model.add(Dense(OUTPUTS_a, activation="softmax"))
    if models == 'CNN':
        model = CNN()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    save_model(model)  # print Summary

    return model


# ------------------------------------------------------------------------------------------------------------------

def train_func(train_ds):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=100)
    check_point = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='accuracy', save_best_only=True)
    final_model = model_definition()
    #
    final_model.fit(train_ds, epochs=n_epoch, callbacks=[early_stop, check_point])


# ------------------------------------------------------------------------------------------------------------------

def predict_func(test_ds):
    final_model = tf.keras.models.load_model('model.h5')
    res = final_model.predict(test_ds)
    xres = [tf.argmax(f).numpy() for f in res]
    data['results'] = xres
    data.to_excel('results.xlsx', index=False)

# ------------------------------------------------------------------------------------------------------------------

def metrics_func(metrics, aggregates=[]):
    '''
    multiple functiosn of metrics to call each function
    f1, cohen, accuracy, mattews correlation
    list of metrics: f1_micro, f1_macro, f1_avg, coh, acc, mat
    list of aggregates : avg, sum
    :return:
    '''

    def f1_score_metric(y_true, y_pred, type):
        '''
            type = micro,macro,weighted,samples
        :param y_true:
        :param y_pred:
        :param average:
        :return: res
        '''
        res = f1_score(y_true, y_pred, average=type)
        print("f1_score {}".format(type), res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('mattews_coef', res)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        print('hamming_loss', res)
        return res

    # For multiclass

    x = lambda x: tf.argmax(x == class_names).numpy()
    y_true = np.array(data['target'].apply(x))
    y_pred = np.array(data['results'])

    # End of Multiclass

    xcont = 1
    xsum = 0
    xavg = 0

    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet = matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet = hamming_metric(y_true, y_pred)
        else:
            xmet = print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum)
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum / xcont)
    # Ask for arguments for each metric


if __name__ == "__main__":
    ### all data
    train_ds = read_train(DATA_DIR)
    test_ds = read_test(DATA_DIR)
    # class_names = process_target(1)  # 1: Multiclass 2: Multilabel 3:Binary
    class_names = train_ds.class_names
    OUTPUTS_a = len(class_names)
    data = show_data(test_ds)
    ## Processing Train dataset

    train_func(train_ds)

    # Preprocessing Test dataset

    predict_func(test_ds)

    ## Metrics Function over the result of the test dataset
    list_of_metrics = ['f1_macro', 'coh', 'acc']
    list_of_agg = ['avg']
    metrics_func(list_of_metrics, list_of_agg)
