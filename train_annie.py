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
# -----------------------
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
# DATA_DIR = OR_PATH + os.path.sep + 'Deep-Learning/animalclassification/Data/'

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
#### Data Augmentation
# ------------------------------------------------------------------------------------------------------------------
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height, img_width, channel)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

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
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

# -----------------------------------------------------------------------------------------------------------------
#### Model for Training
# -----------------------------------------------------------------------------------------------------------------
def model_def():
    # model = Sequential([
    #     data_augmentation,
    #     layers.Rescaling(1. / 255),
    #     layers.Conv2D(16, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(),
    #     layers.Dropout(0.2),
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dense(num_classes)
    # ])

    # Add the pretrained layers
    pretrained_model = keras.applications.ResNet50(include_top=False, weights='imagenet')

    # Add GlobalAveragePooling2D layer
    average_pooling = keras.layers.GlobalAveragePooling2D()(pretrained_model.output)

    # Add the output layer
    output = keras.layers.Dense(12, activation='softmax')(average_pooling)

    # Get the model
    model = keras.Model(inputs=pretrained_model.input, outputs=output)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    save_model(model)
    return model

# -----------------------------------------------------------------------------------------------------------------
def train_func(train_ds):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    check_point = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, mode='min')
    model = model_def()
    history = model.fit(train_ds, validation_data=test_ds, epochs=epochs, callbacks=[early_stop, check_point])
    return history

# -----------------------------------------------------------------------------------------------------------------
### Training + Testing
# -----------------------------------------------------------------------------------------------------------------
data = show_data(test_ds)
class_names = process_target(1)
history = train_func(train_ds)
predict_func(test_ds)
## Metrics Function over the result of the test dataset
list_of_metrics = ['f1_macro', 'coh', 'acc']
list_of_agg = ['avg']
metrics_func(list_of_metrics, list_of_agg)

# -----------------------
### plot

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)
x = np.arange(1, epochs + 1, 1)
fig = plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(x, acc, label='Training Accuracy')
plt.plot(x, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(x)
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()
fig.savefig('plot.pdf', bbox_inches='tight')
