# Importing modules

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import os
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import callbacks
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import glob
import cv2
from tensorflow import random
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout
from keras.applications.xception import Xception
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Sequential, model_from_json, load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


np.random.seed(1)
# Processing training data
# -> appending images in a list 'train_images'
# -> appending labels in a list 'train_labels'

train_images = []
train_labels = []
shape = (224, 224)
train_path = 'DermMel/train_sep/'

for root, dirs, files in os.walk('DermMel/train_sep'):

    for name in dirs:
        # print('folder', name)
        direct = os.path.join(train_path, name)
        # print('direct', direct)
        for filename in os.listdir(direct):
            # print('filename',filename)
            img = cv2.imread(os.path.join(train_path + "/" + name, filename))
            train_labels.append(name)
            img = cv2.resize(img, shape)
            train_images.append(img)
            # print('file', file)

    # for name in files:
    #    print('file', name)
    # if filename.split('.')[1] == 'jpg':
    #    img = cv2.imread(os.path.join(train_path, filename))

    # Spliting file names and storing the labels for image in list
    #    train_labels.append(filename.split('_')[0])

    # Resize all images to a specific shape
    #    img = cv2.resize(img, shape)

    #    train_images.append(img)

# Converting labels into One Hot encoded sparse matrix

train_labels = pd.get_dummies(train_labels).values
print('train_labels shape', train_labels.shape)

# print('train_labels',train_labels)
# Converting train_images to array
train_images = np.array(train_images)
print('train_images shape', train_images.shape)

indices = np.random.randint(0, 1500, 2)
i = 1
plt.figure(figsize=(14, 7))
for each in indices:
    plt.subplot(2, 4, i)
    plt.imshow(train_images[each])
    plt.title(train_labels[each])
    plt.xticks([])
    plt.yticks([])
    i += 1
# print('train_images1',train_images)
# Splitting Training data into train and validation dataset
# x_train,x_val,y_train,y_val = train_test_split(train_images,train_labels,random_state=1)

x_data, x_test, y_data, y_test = train_test_split(train_images, train_labels, test_size=0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
print('X_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('X_val shape  : ', x_val.shape)
print('y_val shape  : ', y_val.shape)
print('X_test shape : ', x_test.shape)
print('y_test shape : ', y_test.shape)
# batch_size = 12
# epochs = 10
base = Xception(include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3))
train_images = base.output
x = GlobalAveragePooling2D()(train_images)
head = Dense(2, activation='softmax')(x)
print('head')
model = Model(inputs=base.input, outputs=head)
print('model')
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
print('compile')
model.summary()
print('summary')
history = model.fit(x_train, y_train, batch_size=12, epochs=10, validation_data=(x_val, y_val))
print('history')
# Processing testing data
# -> appending images in a list 'test_images'
# -> appending labels in a list 'test_labels'
# The test data contains labels as well also we are appending it to a list but we are'nt going to use it while training.
# print('x_train', x_train)
# print('y_train', y_train)
test_images = []
test_labels = []
shape = (224, 224)
test_path = 'DermMel/test/'

for root, dirs, files in os.walk('DermMel/test'):

    for name in dirs:
        # print('folder', name)
        direct = os.path.join(test_path, name)
        # print('direct', direct)
        for filename in os.listdir(direct):
            if filename.split('.')[1] == 'jpg':
                img = cv2.imread(os.path.join(test_path + "/" + name, filename))

                # Spliting file names and storing the labels for image in list
                test_labels.append(name)

                # Resize all images to a specific shape
                img = cv2.resize(img, shape)

                test_images.append(img)

# print('test_images',test_images)
# Converting test_images to array
test_images = np.array(test_images)
# print('test_images1',test_images)
# Visualizing Training data
# print(train_labels[0])
# plt.imshow(train_images[0])
# Visualizing Training data
# print(train_labels[4])
# plt.imshow(train_images[4])
# Creating a Sequential model


# Model Summary
model.summary()
# Training the model
# print('y_val', y_val)
# print('x_val', x_val)
# history = model.fit(x_train,y_train,epochs=10,batch_size=10,validation_data=(x_val,y_val))
# summarize history for accuracy
history_df = pd.DataFrame(history.history)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_df['loss'], label='training loss')
plt.plot(history_df['val_loss'], label='validation loss')
plt.title('Model Loss Function')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_df['accuracy'], label='training accuracy')
plt.plot(history_df['val_accuracy'], label='validation accuracy')
plt.title('Model Accuracy')
plt.legend()
# Predicting labels from X_test data
y_pred = model.predict(x_test)  # Converting prediction classes from one hot encoding to list
# Argmax returns the position of the largest value
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert test labels from one hot encoding to list
y_test_classes = np.argmax(y_test, axis=1)  # Create the confusion matrix
confmx = confusion_matrix(y_test_classes, y_pred_classes)
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confmx, annot=True, fmt='.1f', ax=ax)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
print(classification_report(y_test_classes, y_pred_classes))
test_loss = model.evaluate(x_test, y_test)
test_loss = model.evaluate(x_val, y_val)

# serialize model to JSON
model_json = model.to_json()
with open("melanoma.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("melanoma.h5")
print("Saved model to disk")
# model.save("model.h5")

# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# load weights into new model
# loaded_model.load_weights("modelweight.h5")
# print("Loaded model from disk")

# load model
# model = load_model('model.h5')
# summarize model.
# model.summary()
# evaluate the model set x y value
# evaluate = model.evaluate(x_val,y_val)


# checkImage = test_images[60:61]
# checklabel = test_labels[60:61]
#
# predict = model.predict(np.array(checkImage))
#
# output = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 11: 'A', 12: 'B', 13: 'C',
#           14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J', 21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O',
#           26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T', 31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'}
#
# print("predict :- ", predict)
# print("np.argmax(predict) :- ", np.argmax(predict))
# print("Actual :- ", checklabel)
# print("Predicted :- ", output[np.argmax(predict)])
# checkImage = test_images[1001:1002]
# checklabel = test_labels[1001:1002]
#
# predict = model.predict(np.array(checkImage))
# print("predict :- ", predict)
# print("np.argmax(predict) :- ", np.argmax(predict))
# print("Actual :- ", checklabel)
# print("Predicted :- ", output[np.argmax(predict)])
#
#
# checkImage = test_images[4001:4002]
# checklabel = test_labels[4001:4002]
#
# predict = model.predict(np.array(checkImage))
# print("predict :- ", predict)
# print("np.argmax(predict) :- ", np.argmax(predict))
# print("Actual :- ", checklabel)
# print("Predicted :- ", output[np.argmax(predict)])
#
#
# checkImage = test_images[6001:6002]
# checklabel = test_labels[6001:6002]
#
# predict = model.predict(np.array(checkImage))
# print("predict :- ", predict)
# print("np.argmax(predict) :- ", np.argmax(predict))
# print("Actual :- ", checklabel)
# print("Predicted :- ", output[np.argmax(predict)])
#
#
# checkImage = test_images[8001:8002]
# checklabel = test_labels[8001:8002]
#
# predict = model.predict(np.array(checkImage))
# print("predict :- ", predict)
# print("np.argmax(predict) :- ", np.argmax(predict))
# print("Actual :- ", checklabel)
# print("Predicted :- ", output[np.argmax(predict)])
#
# checkImage = test_images[9001:9002]
# checklabel = test_labels[9001:9002]
#
# predict = model.predict(np.array(checkImage))
# print("predict :- ", predict)
# print("np.argmax(predict) :- ", np.argmax(predict))
# print("Actual :- ", checklabel)
# print("Predicted :- ", output[np.argmax(predict)])
