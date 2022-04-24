import matplotlib.pyplot as plt
import warnings
import numpy as np
import os
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
from PIL import Image
import torch.nn as nn
# -----------------------
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
# DATA_DIR = OR_PATH + os.path.sep + 'Final_project/Data/'

DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)
# -----------------------


img_path = []
img_id = []
png_list = []
for root, animal_folder, img_files in os.walk(DATA_DIR):
    for i in img_files:
        img_path.append(os.path.join(root, i))
for i in img_path:
    img_format = i.split('.')[-1]
    if img_format == 'png':
        png_list.append(i)
for j in png_list:
    im = Image.open(j)
    rgb_im = im.convert('RGB')
    rgb_im.save(f"{j.split('.')[0]}.jpeg")

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
#------------------------------------------------------------------------------------------------------------------
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

#### Data Load
# -----------------------------------------------------------------------------------------------------------------
ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR,
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

data = show_data(ds)
print(data)
print(data.shape)


#### EDA PLOT
Target = data.groupby(['target']).size()
Target.plot.barh(fontsize=20)
plt.ylabel('Animal Class',fontsize=20)
plt.xlabel('Sample counts',fontsize=20)
plt.title("Sample Distribution by Animal Class",fontsize=20)
# As the picture showed, the Animal Dataset is almost balanced.