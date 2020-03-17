from functools import partial

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from utils import *
from metrics import *
from deform_conv.layers import ConvOffset2D


def Conv(input, num_filters, use_deform=False, activation='relu', padding='same', 
         kernel_initializer='he_normal', normal_conv_trainable=True,
         channel_wise=False):
    input = Conv2D(num_filters, (3, 3), activation=activation, padding=padding, 
                   kernel_initializer=kernel_initializer, 
                   trainable=normal_conv_trainable)(input)
    if use_deform:
        input = ConvOffset2D(num_filters, channel_wise=channel_wise)(input)
    
    return input

def Unet(pretrained_weights=None, input_size=(None, None, 3), num_classes=3,
         num_filters=32, use_deform=True, channel_wise=False, 
         normal_conv_trainable=True, class_weights=None):
    
    global Conv
    Conv = partial(Conv, normal_conv_trainable=normal_conv_trainable,
                   use_deform=use_deform, channel_wise=channel_wise)
    
    input = Input(input_size)
    
    conv1 = Conv(input, num_filters, use_deform=False)
    conv1 = Conv(conv1, num_filters, use_deform=False)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv(pool1, num_filters*2)
    conv2 = Conv(conv2, num_filters*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv(pool2, num_filters*4)
    conv3 = Conv(conv3, num_filters*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv(pool3, num_filters*8)
    conv4 = Conv(conv4, num_filters*8)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv(pool4, num_filters*16)
    conv5 = Conv(conv5, num_filters*16)
    drop5 = Dropout(0.5)(conv5)
    
    up6 = Conv(UpSampling2D(size = (2,2))(drop5), num_filters*8)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv(merge6, num_filters*8)
    conv6 = Conv(conv6, num_filters*8)
    
    up7 = Conv(UpSampling2D(size = (2,2))(conv6), num_filters*4)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv(merge7, num_filters*4)
    conv7 = Conv(conv7, num_filters*4)
    
    up8 = Conv(UpSampling2D(size = (2,2))(conv7), num_filters*2)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv(merge8, num_filters*2)
    conv8 = Conv(conv8, num_filters*2)
    
    up9 = Conv(UpSampling2D(size = (2,2))(conv8), num_filters)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv(merge9, num_filters, use_deform=False)
    conv9 = Conv(conv9, num_filters, use_deform=False)
    
    # key_mask = Conv2D(1, 1, activation='sigmoid', name='key_mask', 
    #                   trainable=normal_conv_trainable)(conv9)
    # value_mask = Conv2D(1, 1, activation='sigmoid', name='value_mask', 
    #                     trainable=normal_conv_trainable)(conv9)
    
    output_mask = Conv2D(num_classes, (1, 1), activation='softmax', name='output_mask', 
                         trainable=normal_conv_trainable)(conv9)
    
    model = Model(input=input, outputs=output_mask)
    
    model.compile(optimizer=Adam(lr=1e-4), 
                  loss=partial(custom_categorical_loss, class_weights=class_weights), 
                  metrics=['accuracy', IoU_score])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights, by_name=True)
    
    # model.summary()
    
    return model

# import random
# h = 64
# w = 96
# n = 10
# model = Unet(input_size=(h, w, 1), num_filters=4, use_deform=True)
# 
# X = np.random.rand(n, h, w, 1)
# y = (np.random.rand(n, h, w, 1) > 0.5) * 1.0
# 
# model.fit(X, y, epochs=1)

