# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 21:48:10 2020

@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import shutil
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, BatchNormalization, Input, Conv2DTranspose, concatenate
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split
import random

#%%
loc = os.path.dirname(os.path.realpath(__file__))
os.chdir(loc)


#%%
def trainGenerator(batch_size, arr_img, arr_mask, seed=1):
    

    image_datagen = ImageDataGenerator(rescale = 1/255., rotation_range=5, width_shift_range=0.1, height_shift_range=0.1,
                                       shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode='wrap')


    mask_datagen = ImageDataGenerator(rotation_range=5, width_shift_range=0.1, height_shift_range=0.1,
                                       shear_range=0.05, zoom_range=0.05, horizontal_flip=True, fill_mode='wrap')

    image_gen = image_datagen.flow(arr_img, batch_size=batch_size, shuffle=True, seed=seed )
    mask_gen = mask_datagen.flow(arr_mask, batch_size=batch_size, shuffle=True, seed =seed)
    train_gen = zip(image_gen, mask_gen)
    
    for image, mask in train_gen:
        yield (image, mask)
#%%
def validGenerator(batch_size, arr_img, arr_mask, seed=1):
    

    image_datagen = ImageDataGenerator(rescale = 1/255.)


    mask_datagen = ImageDataGenerator()

    image_gen = image_datagen.flow(arr_img, batch_size=batch_size, shuffle=True, seed=seed )
    mask_gen = mask_datagen.flow(arr_mask, batch_size=batch_size, shuffle=True, seed =seed)
    valid_gen = zip(image_gen, mask_gen)
    
    for image, mask in valid_gen:
        yield (image, mask)


#%%

def get_images(file):
    img = np.load(file)
    k = 0
    temp_img = []
    for i in range(20):
        for j in range(2):
            temp_img.append(img[i, 256*j:256*j+256, 0:256])
            temp_img.append(img[i, 256*j:256*j+256, 195:451])
            temp_img.append(img[i, 256*j:256*j+256, 389:])
    temp_img = np.array(temp_img)
    temp_img = np.expand_dims(temp_img, -1)
    return temp_img


#%%
file_img = 'x_label_manual_20.npy'
file_mask = 'y_label_manual_20.npy'
image_array = get_images(file_img)
mask_array = get_images(file_mask)

#%%
print(image_array.shape, mask_array.shape)

#%%
def data_split(image, mask):
    n = image.shape[0]
    idx = np.arange(n)
    random.shuffle(idx)
    n_train = int(0.8*n)
    n_valid = int(0.1*n)
    train_batch =(image[idx[0:n_train]], mask[idx[0:n_train]])
    valid_batch = (image[idx[n_train:n_train+n_valid]], mask[idx[n_train:n_train+n_valid]])
    test_batch = (image[idx[n_train+n_valid:]], mask[idx[n_train+n_valid:]])
    return train_batch, valid_batch, test_batch


#%%
train_batch, valid_batch, test_batch = data_split(image_array, mask_array)

#%%
print(train_batch[0].shape, train_batch[1].shape)

#%%
batch_size=8
train_generator = trainGenerator(batch_size, train_batch[0], train_batch[1])
valid_generator = validGenerator(batch_size, valid_batch[0], valid_batch[1])
#%%
i = 0
for batch in valid_generator:
    img, mask = batch
    print(img.shape, mask.shape)
    i+=1
    if i>3:
        break
    
#%%
def create_model():
    inputs = Input((256, 256, 1))
    
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPool2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPool2D((2, 2)) (c2)
    
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPool2D((2, 2)) (c3)
    
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPool2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
#%%
model = create_model()
#%%
loss = 'binary_crossentropy'
opti = Adam(learning_rate=0.0005)
metrics=['accuracy']
model.compile(optimizer=opti, loss=loss, metrics=metrics)

#%%
early_stopping = EarlyStopping(patience=3)
callbacks = ModelCheckpoint('steel_seg.h5', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')
model.fit_generator(train_generator, validation_data=valid_generator, validation_steps=10, steps_per_epoch=150, epochs=5,
                    callbacks=[early_stopping, callbacks])

#%%
x = test_batch[0] / 255.
y = test_batch[1]
ypred = model.predict(x)
#%%
for i in range(12):
    temp = (ypred[i] > 0.5).astype(np.uint8)
    temp_plot = np.concatenate((y[i,:,:,0],np.ones((y.shape[1], 20)), temp[:,:,0]), axis=1)
    plt.imshow(temp_plot)
    #plt.savefig('results/test/'+str(i+12)+'.jpg')
    plt.show()
    
#%%
model.save('final_model.h5')

#%%
def mean_IOU(yact, ypred):
    ypred = (ypred>0.5).astype('uint8')
    
    inter = np.sum(yact*ypred)
    union = np.sum(yact) + np.sum(ypred) - inter
    return inter / (union+0.005)
def dice_coef(yact, ypred):
    ypred = (ypred>0.5).astype('uint8')
    
    inter = np.sum(yact*ypred)
    union = np.sum(yact) + np.sum(ypred) - inter
    return 2*inter / (inter + union + 0.005)    
    
    
    
iou = mean_IOU(y, ypred)
dice_coef = dice_coef(y, ypred)
print('IOU is :', iou)
print('Dice coef is: ', dice_coef)
    
    
