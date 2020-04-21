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
   
    def conv_act_bn_dropout_block(input, num_filters, use_deform=True,
                                  use_dropout=False):
        output = Conv(input, num_filters, use_deform=True)
        output = Activation('relu')(output)
        output = BatchNormalization()(output)
        if use_dropout:
            output = SpatialDropout2D(0.2)(output)
        
        return output
    
    input = Input(input_size)
    
    conv1 = conv_act_bn_dropout_block(input, num_filters, use_deform=True)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=True)
    down1_2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_act_bn_dropout_block(down1_2, num_filters*2, use_deform=True)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=True)
    down2_3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_act_bn_dropout_block(down2_3, num_filters*4, use_deform=True)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=True)
    down3_4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_act_bn_dropout_block(down3_4, num_filters*8, use_deform=True)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=True)
    down4_5 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = conv_act_bn_dropout_block(down4_5, num_filters*16, use_deform=True)
    conv5 = conv_act_bn_dropout_block(conv5, num_filters*16, use_deform=True)
    # down5_6 = MaxPooling2D(pool_size=(2, 2))(conv5)
    # 
    # conv6 = conv_act_bn_dropout_block(down5_6, num_filters*32, use_deform=True)
    # conv6 = conv_act_bn_dropout_block(conv6, num_filters*32, use_deform=True)
    # 
    # up6_5 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv6), num_filters*16)
    # merge5 = concatenate([conv5, up6_5], axis=3)
    # conv5 = conv_act_bn_dropout_block(merge5, num_filters*16, use_deform=True)
    # conv5 = conv_act_bn_dropout_block(conv5, num_filters*16, use_deform=True)
    
    up5_4 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv5), num_filters*8)
    merge4 = concatenate([conv4, up5_4], axis=3)
    conv4 = conv_act_bn_dropout_block(merge4, num_filters*8, use_deform=True)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=True)
    
    up4_3 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv4), num_filters*4)
    merge3 = concatenate([conv3, up4_3], axis=3)
    conv3 = conv_act_bn_dropout_block(merge3, num_filters*4, use_deform=True)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=True)
    
    up3_2 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv3), num_filters*2)
    merge2 = concatenate([conv2, up3_2], axis=3)
    conv2 = conv_act_bn_dropout_block(merge2, num_filters*2, use_deform=True)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=True)
    
    up2_1 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv2), num_filters)
    merge1 = concatenate([conv1, up2_1], axis=3)
    conv1 = conv_act_bn_dropout_block(merge1, num_filters, use_deform=False)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=False)
    
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