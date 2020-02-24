import keras
from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

from deform_conv.layers import ConvOffset2D

def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def custom_loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    cross_entropy = keras.losses.binary_crossentropy(y_true, y_pred)
    return 5 * dice + cross_entropy

def IoU_score(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def Conv(input, num_filters, use_deform=False, activation='relu', padding='same', 
         kernel_initializer='he_normal'):
    input = Conv2D(num_filters, (3, 3), activation=activation, padding=padding, 
                   kernel_initializer=kernel_initializer)(input)
    if use_deform:
        input = ConvOffset2D(num_filters)(input)
    
    return input

def Unet(pretrained_weights=None, input_size=(None, None, 3), num_filters=32, 
         use_deform=True):
    
    inputs = Input(input_size)
    
    input = Input(input_size)
    
    conv1 = Conv(input, num_filters, use_deform=False)
    conv1 = Conv(conv1, num_filters, use_deform=False)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv(pool1, num_filters*2, use_deform)
    conv2 = Conv(conv2, num_filters*2, use_deform)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv(pool2, num_filters*4, use_deform)
    conv3 = Conv(conv3, num_filters*4, use_deform)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv(pool3, num_filters*8, use_deform)
    conv4 = Conv(conv4, num_filters*8, use_deform)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv(pool4, num_filters*16, use_deform)
    conv5 = Conv(conv5, num_filters*16, use_deform)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv(UpSampling2D(size = (2,2))(drop5), num_filters*8, use_deform)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv(merge6, num_filters*8, use_deform)
    conv6 = Conv(conv6, num_filters*8, use_deform)
    
    up7 = Conv(UpSampling2D(size = (2,2))(conv6), num_filters*4, use_deform)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv(merge7, num_filters*4, use_deform)
    conv7 = Conv(conv7, num_filters*4, use_deform)
    
    up8 = Conv(UpSampling2D(size = (2,2))(conv7), num_filters*2, use_deform)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv(merge8, num_filters*2, use_deform)
    conv8 = Conv(conv8, num_filters*2, use_deform)
    
    up9 = Conv(UpSampling2D(size = (2,2))(conv8), num_filters, use_deform)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv(merge9, num_filters, use_deform=False)
    conv9 = Conv(conv9, num_filters, use_deform=False)
    
    output = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    model = Model(input=input, output=output)
    
    model.compile(optimizer=Adam(lr=1e-4), loss=custom_loss, metrics=['accuracy', IoU_score])
    
    model.summary()

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

h = 64
w = 96
n = 10
model = Unet(input_size=(h, w, 1), num_filters=4, use_deform=False)

import numpy as np
X = np.random.rand(n, h, w, 1)
y = (np.random.rand(n, h, w, 1) > 0.5) * 1.0

model.fit(X, y)