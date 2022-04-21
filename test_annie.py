import matplotlib.pyplot as plt
import warnings
import numpy as np
import os
import PIL
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
import pathlib
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef

# -----------------------
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
# DATA_DIR = OR_PATH + os.path.sep + 'Deep-Learning/animalclassification/Data/'

DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)
# -----------------------
# -----------------------
random_seed = 42
train_size = 0.8

batch_size = 64
epochs = 20
lr = 0.01
img_height = 224
img_width = 224
channel = 3
# -----------------------
# ------------------------------------------------------------------------------------------------------------------
#### def
# -------------------------------------------------------------------------------------------------------------------
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


# -------------------------------------------------------------------------------------------------------------------

def process_target(target_type):
    '''
        1- Multiclass  target = (1...n, text1...textn)
        2- Multilabel target = ( list(Text1, Text2, Text3 ) for each observation, separated by commas )
        3- Binary   target = (1,0)

    :return:
    '''
    class_names = np.sort(data['target'].unique())

    if target_type == 1:

        x = lambda x: tf.argmax(x == class_names).numpy()

        final_target = data['target'].apply(x)
        data['true'] = final_target

        final_target = to_categorical(list(final_target))

        xfinal = []

        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal

        data['target_class'] = final_target

# -------------------------------------------------------------------------------------------------------------------
def save_model(model):
    '''
       receives the model and print the summary into a .txt file
  '''
    with open('model_summary.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

# -----------------------------------------------------------------------------------------------------------------
def predict_func(test_ds):
    final_model = tf.keras.models.load_model('model.h5')
    res = final_model.predict(test_ds)
    xres = [tf.argmax(f).numpy() for f in res]
    loss, accuracy = final_model.evaluate(test_ds)
    data['results'] = xres
    data.to_excel('results.xlsx', index=False)

# -----------------------------------------------------------------------------------------------------------------

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

    # For multiclass

    y_true = np.array(data['true'])
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
        else:
            xmet = print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum)
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum / xcont)

# -----------------------------------------------------------------------------------------------------------------
#### Data Load
# -----------------------------------------------------------------------------------------------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# -----------------------------------------------------------------------------------------------------------------
#### Testing
# -----------------------------------------------------------------------------------------------------------------
data = show_data(test_ds)
class_names = process_target(1)
predict_func(test_ds)
## Metrics Function over the result of the test dataset
list_of_metrics = ['f1_macro', 'coh', 'acc']
list_of_agg = ['avg']
metrics_func(list_of_metrics, list_of_agg)
