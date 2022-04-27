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
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.efficientnet import EfficientNet
import pathlib
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
from PIL import Image
from tensorflow.python.keras.layers import Flatten, Dense

# -----------------------
"""""
pip install split-folders
"""""
import os
import splitfolders
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

random_seed = 42
inputfolder = DATA_DIR

splitfolders.ratio(inputfolder, output='train_test', seed=random_seed, ratio=(0.9,0,0.1),group_prefix=None)

# -----------------------
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR_test = os.getcwd() + os.path.sep + 'Code' + os.path.sep +'train_test' + os.path.sep + 'test'
DATA_DIR_train = os.getcwd() + os.path.sep + 'Code' + os.path.sep +'train_test' + os.path.sep + 'train'
sep = os.path.sep
os.chdir(OR_PATH)
# -----------------------
# -----------------------
random_seed = 42
train_size = 0.8

batch_size = 64
epochs = 3
lr = 0.01
img_height = 224
img_width = 224
channel = 3
# -----------------------
# -----------------------------------------------------------------------------------------------------------------
#### Data Load
# -----------------------------------------------------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR_train,
    validation_split=0.1,
    subset="training",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR_train,
    validation_split=0.1,
    subset="validation",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)



class_names = train_ds.class_names
num_classes = len(class_names)

# -----------------------------------------------------------------------------------------------------------------
#### Data Augmentation plot
# -----------------------------------------------------------------------------------------------------------------
def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)
  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
  plt.tight_layout()

def flip(data):
    data_flipped = tf.image.flip_left_right(data)
    return data_flipped
def rotation(data):
    data_rotated = tf.image.rot90(data)
    return data_rotated

input_size = [img_height, img_width]
def resize(data):
    data_resized = tf.image.resize(data, input_size)
    return data_resized

for images, labels in train_ds.take(1):
    img = images[1].numpy().astype("uint8")
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(class_names[labels[1]], fontdict={'fontsize':12})
    plt.axis("off")
    plt.show()
    plt.figure(figsize=(12, 8))
    visualize(img, flip(img))
    plt.show()
    plt.figure(figsize=(12, 8))
    visualize(img, rotation(img))
    plt.show()

# ------------------------------------------------------------------------------------------------------------------
#### def
# ------------------------------------------------------------------------------------------------------------------
def all_data(file_dir):
    img_path = []
    img_id = []
    for root, sub_folders, files in os.walk(file_dir):
        for i in files:
            img_path.append(os.path.join(root, i))
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
    data = np.array([img_path, img_id, labels])
    data = data.transpose()
    path_list = list(data[:, 0])
    id_list = list(data[:, 1])
    label_list = list(data[:, 2])
    df = pd.DataFrame((path_list, id_list, label_list)).T
    df.rename(columns={0:'path', 1:"id", 2: 'target'}, inplace=True)
    return df


# -------------------------------------------------------------------------------------------------------------------
def process_target(target_type):
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
    return class_names


# -------------------------------------------------------------------------------------------------------------------
def process_path(feature, target):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''
    label = target
    file_path = feature
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=channel)
    img = tf.image.resize(img, [img_height, img_width])
    return img, label


# -------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------
def read_data():

    ds_inputs = np.array(DATA_DIR + sep +data['id'])
    ds_targets = np.array(data['true'])

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) # creates a tensor from the image paths and targets

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(batch_size)

    return final_ds


# -----------------------------------------------------------------------------------------------------------------
def predict_func(test_ds):
    final_model = tf.keras.models.load_model('model_VGG19.h5')
    res = final_model.predict(test_ds)
    xres = [tf.argmax(f).numpy() for f in res]
    loss, accuracy = final_model.evaluate(test_ds)
    data['results'] = xres
    data.to_excel('results_VGG19.xlsx', index=False)


# -----------------------------------------------------------------------------------------------------------------
def metrics_func(metrics, aggregates=[]):
    def f1_score_metric(y_true, y_pred, type):
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
        return res

    # For multiclass

    y_true = np.array(data['true'])
    y_pred = np.array(data['results'])

    # End of Multiclass

    xcont = 0
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
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum)
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum / xcont)

# -----------------------------------------------------------------------------------------------------------------
### Model
# -----------------------------------------------------------------------------------------------------------------
pretrained_model = keras.applications.ResNet50(include_top=False, weights='imagenet')
# Add GlobalAveragePooling2D layer
average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)
# Add the output layer
output = keras.layers.Dense(12, activation='softmax')(average_pooling)
model = keras.Model(inputs=pretrained_model.input, outputs=output)
for layer in pretrained_model.layers:
    # Freeze the layer
    layer.trainable = False
check_point = keras.callbacks.ModelCheckpoint('model_ResNet50.h5', save_best_only=True,
                                                          save_weights_only=True, monitor='val_loss')
    # EarlyStopping callback
early_stop = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True, monitor='val_loss')
    # ReduceLROnPlateau callback
reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=1)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop, check_point, reduce_lr_on_plateau])
for layer in pretrained_model.layers:
    # Unfreeze the layer
    layer.trainable = True
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_ds, epochs=3, validation_data=val_ds, callbacks=[early_stop, check_point, reduce_lr_on_plateau])

# -----------------------------------------------------------------------------------------------------------------
### Testing
# -----------------------------------------------------------------------------------------------------------------
data = all_data(DATA_DIR)
class_names= process_target(1)
test_ds = read_data()
loss, accuracy = model.evaluate(test_ds)
predict_func(test_ds)
list_of_metrics = ['f1_micro', 'coh', 'acc']
list_of_agg = ['avg']
metrics_func(list_of_metrics, list_of_agg)

