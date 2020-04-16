from functools import partial

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import regularizers

from utils import *
from metrics import *
from deform_conv.layers import ConvOffset2D


def Conv(input, num_filters, use_deform=False, activation='relu', padding='same', 
         kernel_initializer='he_normal', normal_conv_trainable=True,
         channel_wise=False):
    input = Conv2D(num_filters, (3, 3), activation=None, padding=padding, 
                   kernel_initializer=kernel_initializer, 
                   kernel_regularizer=regularizers.l2(0.01),
                   trainable=normal_conv_trainable)(input)
    if use_deform:
        input = ConvOffset2D(num_filters, channel_wise=channel_wise)(input)
    
    return input

def Unet(pretrained_weights=None, input_size=(None, None, 3), num_classes=3,
         num_filters=32, use_deform=True, channel_wise=False, 
         normal_conv_trainable=True, class_weights=None, loss_weights=[1.0, 1.0],
         ignore_background=False):
    
    global Conv
    Conv = partial(Conv, normal_conv_trainable=normal_conv_trainable,
                   use_deform=use_deform, channel_wise=channel_wise)
    
    input = Input(input_size)
    
    conv1 = Conv(input, num_filters, use_deform=True)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv(conv1, num_filters, use_deform=True)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    down1_2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv(down1_2, num_filters*2, use_deform=True)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv(conv2, num_filters*2, use_deform=True)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    down2_3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv(down2_3, num_filters*4, use_deform=True)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv(conv3, num_filters*4, use_deform=True)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    down3_4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv(down3_4, num_filters*8)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv(conv4, num_filters*8)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    # drop4 = Dropout(0.5)(conv4)
    down4_5 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv(down4_5, num_filters*16)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv(conv5, num_filters*16)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    # drop5 = Dropout(0.5)(conv5)
    down5_6 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv6 = Conv(down5_6, num_filters*32)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv(conv6, num_filters*32)
    conv6 = Activation('relu')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    up6_5 = Conv(UpSampling2D(size = (2,2))(conv6), num_filters*16)
    merge5 = concatenate([conv5, up6_5], axis=3)
    conv5 = Conv(merge5, num_filters*16, use_deform=True)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv(conv5, num_filters*16, use_deform=True)
    conv5 = Activation('relu')(conv5)
    conv5 = BatchNormalization()(conv5)
    
    up5_4 = Conv(UpSampling2D(size = (2,2))(conv5), num_filters*8)
    merge4 = concatenate([conv4, up5_4], axis=3)
    conv4 = Conv(merge4, num_filters*8, use_deform=True)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv(conv4, num_filters*8, use_deform=True)
    conv4 = Activation('relu')(conv4)
    conv4 = BatchNormalization()(conv4)
    
    up4_3 = Conv(UpSampling2D(size = (2,2))(conv4), num_filters*4)
    merge3 = concatenate([conv3, up4_3], axis=3)
    conv3 = Conv(merge3, num_filters*4, use_deform=True)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv(conv3, num_filters*4, use_deform=True)
    conv3 = Activation('relu')(conv3)
    conv3 = BatchNormalization()(conv3)
    
    up3_2 = Conv(UpSampling2D(size = (2,2))(conv3), num_filters*2)
    merge2 = concatenate([conv2, up3_2], axis=3)
    conv2 = Conv(merge2, num_filters*2, use_deform=True)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv(conv2, num_filters*2, use_deform=True)
    conv2 = Activation('relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    
    up2_1 = Conv(UpSampling2D(size = (2,2))(conv2), num_filters)
    merge1 = concatenate([conv1, up2_1], axis=3)
    conv1 = Conv(merge1, num_filters, use_deform=False)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv(conv1, num_filters, use_deform=False)
    conv1 = Activation('relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    
    # key_mask = Conv2D(1, 1, activation='sigmoid', name='key_mask', 
    #                   trainable=normal_conv_trainable)(conv9)
    # value_mask = Conv2D(1, 1, activation='sigmoid', name='value_mask', 
    #                     trainable=normal_conv_trainable)(conv9)
    
    output_mask = Conv2D(num_classes, (1, 1), activation='softmax', name='output_mask', 
                         trainable=normal_conv_trainable)(conv1)
    
    model = Model(input=input, outputs=output_mask)
    
    global IoU_score
    IoU_score = partial(IoU_score, ignore_last_channel=ignore_background)
    IoU_score.__name__ = 'IoU_score'
    
    global custom_categorical_loss
    custom_categorical_loss = partial(custom_categorical_loss, 
                                      class_weights=class_weights,
                                      loss_weights=loss_weights,
                                      ignore_last_channel=ignore_background)
    custom_categorical_loss.__name__ = 'custom_categorical_loss'
    
    model.compile(optimizer=Adam(lr=1e-4), 
                  loss=custom_categorical_loss, 
                  metrics=['accuracy', IoU_score])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights, by_name=True)
    
    # model.summary()
    
    return model

# import random
# h = 1024
# w = 768
# n = 1
# c = 3
# model = Unet(input_size=(h, w, 3), num_filters=1, use_deform=True)
# 
# X = np.random.rand(n, h, w, 3)
# y = (np.random.rand(n, h, w, 3) > 0.5) * 1.0
# 
# model.fit(X, y, epochs=1)

