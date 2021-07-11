Created on Tue Jul 14 00:29:27 2020
################################################################
################################  Augmented model ##############
################################################################
@author: ALIReza Javanmardi
"""
!unzip -uq "/content/drive/My Drive/cv_project/RGB-faces-128x128.zip"

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Dense,Dropout,Flatten,BatchNormalization,LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16

import numpy as np
from sklearn.model_selection import KFold
import os
import cv2
import re
import matplotlib.pyplot as plt


images_path = "/content/RGB-faces-128x128/"

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
dirlist = sorted_alphanumeric(os.listdir(images_path))
x = []
for i in dirlist:
    face = cv2.imread('/content/RGB-faces-128x128/'+i)
    x.append(face)
x = np.array(x)
num_faces = np.size(dirlist)
## detecting classes less than 14 face 
## which are 11,47,48,50,73,74,76,82,96
s=[]
a=1
for q in range(1531):
    res = [int(i) for i in dirlist[q].split('-') if i.isdigit()]
    res_new = [int(i) for i in dirlist[q+1].split('-') if i.isdigit()]
    if res[0] == res_new[0]:
        a=a+1
    elif a==14:
        s.append(res[0])
        a=1
    elif res[0] != res_new[0]:
        a=1

################# producing labels#################
        
label = np.ones((num_faces,),dtype=int)
a = 0
for j in range(0,10):
    label[a:a+14]=(j)
    a=a+14
label[140:153]=10
a = 153
for j in range(11,46):
    label[a:a+14]=j
    a=a+14
label[643:656]=47
label[656:670]=48
label[670:679]=49
a = 679
for j in range(50,72):
    label[a:a+14]=j
    a=a+14
label[987:1000]=72
label[1000:1005]=73
label[1005:1019]=74
label[1019:1024]=75
a = 1024
for j in range(76,81):
    label[a:a+14]=j
    a=a+14
label[1094:1107]=81
a = 1107
for j in range(82,95):
    label[a:a+14]=j
    a=a+14
label[1289:1294]=95
a = 1294
for j in range(96,113):
    label[a:a+14]=j
    a=a+14


(x,y) = (x,label)

y = to_categorical(y,113)

x = x.astype('float32')

x /=255

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
datagen = ImageDataGenerator(horizontal_flip=True)

for train, test in kfold.split(x,y):



  ############## Model ##################
  base_model = VGG16(weights='imagenet',include_top=False,input_shape=(128,128,3))
  base_model.trainable = False
  base_model.get_layer('block5_conv1').trainable = True
  base_model.get_layer('block5_conv2').trainable = True
  base_model.get_layer('block5_conv3').trainable = True

  input_layer = base_model.output
  flatten_layer = Flatten()(input_layer)
  z=Dense(units = 200 , activation = 'relu')(flatten_layer)
  z = Dropout(rate = 0.5)(z)
  output_layer=Dense(units = 113 , activation = 'softmax')(z)
  model = Model(base_model.input,output_layer)

  opt = Adam(lr = 0.00001)
  #model.summary()
  model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics=['accuracy'])

  datagen.fit(x[train])
# fits the model on batches with real-time data augmentation:
  history = model.fit(datagen.flow(x[train], y[train], batch_size=128),
          steps_per_epoch=len(x[train]) // 128, epochs=100,shuffle=True)

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  # Generate generalization metrics
  scores = model.evaluate(x[test],y[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 00:29:27 2020

@author: ALI
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,Dense,Dropout,Flatten,BatchNormalization,LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.densenet import preprocess_input

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import cv2
import re
import matplotlib.pyplot as plt


images_path = "/content/RGB-faces-128x128/"

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)
dirlist = sorted_alphanumeric(os.listdir(images_path))
x = []
for i in dirlist:
    face = cv2.imread('/content/RGB-faces-128x128/'+i)
    x.append(face)
x = np.array(x)
num_faces = np.size(dirlist)
## detecting classes less than 14 face 
## which are 11,47,48,50,73,74,76,82,96
s=[]
a=1
for q in range(1531):
    res = [int(i) for i in dirlist[q].split('-') if i.isdigit()]
    res_new = [int(i) for i in dirlist[q+1].split('-') if i.isdigit()]
    if res[0] == res_new[0]:
        a=a+1
    elif a==14:
        s.append(res[0])
        a=1
    elif res[0] != res_new[0]:
        a=1

################# producing labels#################
        
label = np.ones((num_faces,),dtype=int)
a = 0
for j in range(0,10):
    label[a:a+14]=(j)
    a=a+14
label[140:153]=10
a = 153
for j in range(11,46):
    label[a:a+14]=j
    a=a+14
label[643:656]=47
label[656:670]=48
label[670:679]=49
a = 679
for j in range(50,72):
    label[a:a+14]=j
    a=a+14
label[987:1000]=72
label[1000:1005]=73
label[1005:1019]=74
label[1019:1024]=75
a = 1024
for j in range(76,81):
    label[a:a+14]=j
    a=a+14
label[1094:1107]=81
a = 1107
for j in range(82,95):
    label[a:a+14]=j
    a=a+14
label[1289:1294]=95
a = 1294
for j in range(96,113):
    label[a:a+14]=j
    a=a+14

##################  shuffling the data ##################

data , label = shuffle(x,label,random_state = 2)
train_data = [data,label]

############# training process ##################

batch_size = 32
(x,y) = (train_data[0],train_data[1])
x_train,x_test , y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 2)

y_train = to_categorical(y_train,113)
y_test = to_categorical(y_test,113)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /=255
x_test /=255



############## Model ##################

base_model = VGG16(weights='imagenet',include_top=False,input_shape=(128,128,3))
base_model.trainable = False
base_model.get_layer('block5_conv1').trainable = True
base_model.get_layer('block5_conv2').trainable = True
base_model.get_layer('block5_conv3').trainable = True

input_layer = base_model.output
flatten_layer = Flatten()(input_layer)
z=Dense(units = 200 , activation = 'relu')(flatten_layer)
z = Dropout(rate = 0.5)(z)
output_layer=Dense(units = 113 , activation = 'softmax')(z)
model = Model(base_model.input,output_layer)

opt = Adam(lr = 0.00001)
model.summary()
model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics=['accuracy'])

hist = model.fit(x_train,y_train, batch_size=128, epochs = 200, shuffle=True, validation_split=0.2)
result = model.evaluate(x_test , y_test)
print(result[0])
print(result[1])

history = hist.history

losses = history['loss']
val_loss = history['val_loss']
accuracies = history['accuracy']
val_accuracies = history['val_accuracy']
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses)
plt.plot(val_loss)

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(accuracies)
plt.plot(val_accuracies)

!unzip -uq "/content/drive/My Drive/cv_project/RGB-faces-128x128.zip"
