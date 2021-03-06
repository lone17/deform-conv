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

def text_classification_model(pretrained_weights=None, input_size=(None, None, 3), 
              num_classes=3, num_filters=32, use_deform=True, 
              channel_wise=False, normal_conv_trainable=True, 
              class_weights=None, loss_weights=[1.0, 1.0], 
              ignore_background=False):
    
    global Conv
    Conv = partial(Conv, normal_conv_trainable=normal_conv_trainable,
                   channel_wise=channel_wise)
   
    def conv_act_bn_dropout_block(input, num_filters, use_deform=True,
                                  dropout=0):
        output = Conv(input, num_filters, use_deform=use_deform)
        output = Activation('relu')(output)
        output = BatchNormalization()(output)
        if dropout > 0:
            output = SpatialDropout2D(dropout)(output)
        
        return output
    
    input = Input(input_size)
    
    conv1 = conv_act_bn_dropout_block(input, num_filters, use_deform=use_deform)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=use_deform)
    down1_2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_act_bn_dropout_block(down1_2, num_filters*2, use_deform=use_deform)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=use_deform)
    down2_3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_act_bn_dropout_block(down2_3, num_filters*4, use_deform=use_deform)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=use_deform)
    down3_4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_act_bn_dropout_block(down3_4, num_filters*8, use_deform=use_deform)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=use_deform)
    down4_5 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = conv_act_bn_dropout_block(down4_5, num_filters*16, use_deform=use_deform)
    conv5 = conv_act_bn_dropout_block(conv5, num_filters*16, use_deform=use_deform)
    # down5_6 = MaxPooling2D(pool_size=(2, 2))(conv5)
    # 
    # conv6 = conv_act_bn_dropout_block(down5_6, num_filters*32, use_deform=use_deform)
    # conv6 = conv_act_bn_dropout_block(conv6, num_filters*32, use_deform=use_deform)
    # 
    # up6_5 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv6), num_filters*16)
    # merge5 = concatenate([conv5, up6_5], axis=3)
    # conv5 = conv_act_bn_dropout_block(merge5, num_filters*16, use_deform=use_deform)
    # conv5 = conv_act_bn_dropout_block(conv5, num_filters*16, use_deform=use_deform)
    
    up5_4 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv5), num_filters*8)
    merge4 = concatenate([conv4, up5_4], axis=3)
    conv4 = conv_act_bn_dropout_block(merge4, num_filters*8, use_deform=use_deform)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=use_deform)
    
    up4_3 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv4), num_filters*4)
    merge3 = concatenate([conv3, up4_3], axis=3)
    conv3 = conv_act_bn_dropout_block(merge3, num_filters*4, use_deform=use_deform)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=use_deform)
    
    up3_2 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv3), num_filters*2)
    merge2 = concatenate([conv2, up3_2], axis=3)
    conv2 = conv_act_bn_dropout_block(merge2, num_filters*2, use_deform=use_deform)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=use_deform)
    
    up2_1 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv2), num_filters)
    merge1 = concatenate([conv1, up2_1], axis=3)
    conv1 = conv_act_bn_dropout_block(merge1, num_filters, use_deform=False)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=False)
    
    # key_mask = Conv2D(1, 1, activation='sigmoid', name='key_mask', 
    #                   trainable=normal_conv_trainable)(conv9)
    # value_mask = Conv2D(1, 1, activation='sigmoid', name='value_mask', 
    #                     trainable=normal_conv_trainable)(conv9)
    
    output_mask = Conv2D(num_classes, (1, 1), activation='softmax', 
                         name='output_mask', 
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

def relation_model(pretrained_weights=None, input_size=(None, None, 3), 
                  num_filters=32, use_deform=True, 
                  channel_wise=False, normal_conv_trainable=True, 
                  loss_weights=[1.0, 1.0]):
    
    global Conv
    Conv = partial(Conv, normal_conv_trainable=normal_conv_trainable,
                   use_deform=use_deform, channel_wise=channel_wise)
   
    def conv_act_bn_dropout_block(input, num_filters, use_deform=True,
                                  dropout=0):
        output = Conv(input, num_filters, use_deform=use_deform)
        output = Activation('relu')(output)
        output = BatchNormalization()(output)
        if dropout > 0:
            output = SpatialDropout2D(dropout)(output)
        
        return output
    
    input = Input(input_size)
    
    conv1 = conv_act_bn_dropout_block(input, num_filters, use_deform=use_deform)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=use_deform)
    down1_2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_act_bn_dropout_block(down1_2, num_filters*2, use_deform=use_deform)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=use_deform)
    down2_3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_act_bn_dropout_block(down2_3, num_filters*4, use_deform=use_deform)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=use_deform)
    down3_4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_act_bn_dropout_block(down3_4, num_filters*8, use_deform=use_deform)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=use_deform)
    down4_5 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = conv_act_bn_dropout_block(down4_5, num_filters*16, use_deform=use_deform)
    conv5 = conv_act_bn_dropout_block(conv5, num_filters*16, use_deform=use_deform)
    # down5_6 = MaxPooling2D(pool_size=(2, 2))(conv5)
    # 
    # conv6 = conv_act_bn_dropout_block(down5_6, num_filters*32, use_deform=use_deform)
    # conv6 = conv_act_bn_dropout_block(conv6, num_filters*32, use_deform=use_deform)
    # 
    # up6_5 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv6), num_filters*16)
    # merge5 = concatenate([conv5, up6_5], axis=3)
    # conv5 = conv_act_bn_dropout_block(merge5, num_filters*16, use_deform=use_deform)
    # conv5 = conv_act_bn_dropout_block(conv5, num_filters*16, use_deform=use_deform)
    
    up5_4 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv5), num_filters*8)
    merge4 = concatenate([conv4, up5_4], axis=3)
    conv4 = conv_act_bn_dropout_block(merge4, num_filters*8, use_deform=use_deform)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=use_deform)
    
    up4_3 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv4), num_filters*4)
    merge3 = concatenate([conv3, up4_3], axis=3)
    conv3 = conv_act_bn_dropout_block(merge3, num_filters*4, use_deform=use_deform)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=use_deform)
    
    up3_2 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv3), num_filters*2)
    merge2 = concatenate([conv2, up3_2], axis=3)
    conv2 = conv_act_bn_dropout_block(merge2, num_filters*2, use_deform=use_deform)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=use_deform)
    
    up2_1 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv2), num_filters)
    merge1 = concatenate([conv1, up2_1], axis=3)
    conv1 = conv_act_bn_dropout_block(merge1, num_filters, use_deform=False)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=False)
    
    horizontal_relation_mask = Conv2D(1, 1, activation='sigmoid', 
                                      name='horizontal_relation_mask')(conv1)
    
    vertical_relation_mask = Conv2D(1, 1, activation='sigmoid', 
                                    name='vertical_relation_mask')(conv1)
    
    model = Model(input=input, 
                  outputs=[horizontal_relation_mask, vertical_relation_mask])
    
    global IoU_score
    IoU_score = partial(IoU_score, ignore_last_channel=False)
    IoU_score.__name__ = 'IoU_score'
    
    global custom_loss
    custom_loss = partial(custom_loss, class_weights=None, 
                          loss_weights=loss_weights, ignore_last_channel=False)
    custom_loss.__name__ = 'custom_loss'
    
    model.compile(optimizer=Adam(lr=1e-4), 
                  loss={'horizontal_relation_mask': custom_loss,
                        'vertical_relation_mask': custom_loss}, 
                  metrics=['accuracy', IoU_score])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights, by_name=True)
    
    # model.summary()
    
    return model

def text_detection_model(pretrained_weights=None, input_size=(None, None, 3), 
                  num_filters=32, use_deform=True, 
                  channel_wise=False, normal_conv_trainable=True, 
                  loss_weights=[1.0, 1.0]):
    
    global Conv
    Conv = partial(Conv, normal_conv_trainable=normal_conv_trainable,
                   use_deform=use_deform, channel_wise=channel_wise)
   
    def conv_act_bn_dropout_block(input, num_filters, use_deform=True,
                                  dropout=0):
        output = Conv(input, num_filters, use_deform=use_deform)
        output = Activation('relu')(output)
        output = BatchNormalization()(output)
        if dropout > 0:
            output = SpatialDropout2D(dropout)(output)
        
        return output
    
    input = Input(input_size)
    
    conv1 = conv_act_bn_dropout_block(input, num_filters, use_deform=use_deform)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=use_deform)
    down1_2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_act_bn_dropout_block(down1_2, num_filters*2, use_deform=use_deform)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=use_deform)
    down2_3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_act_bn_dropout_block(down2_3, num_filters*4, use_deform=use_deform)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=use_deform)
    down3_4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_act_bn_dropout_block(down3_4, num_filters*8, use_deform=use_deform)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=use_deform)
    down4_5 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = conv_act_bn_dropout_block(down4_5, num_filters*16, use_deform=use_deform)
    conv5 = conv_act_bn_dropout_block(conv5, num_filters*16, use_deform=use_deform)
    # down5_6 = MaxPooling2D(pool_size=(2, 2))(conv5)
    # 
    # conv6 = conv_act_bn_dropout_block(down5_6, num_filters*32, use_deform=use_deform)
    # conv6 = conv_act_bn_dropout_block(conv6, num_filters*32, use_deform=use_deform)
    # 
    # up6_5 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv6), num_filters*16)
    # merge5 = concatenate([conv5, up6_5], axis=3)
    # conv5 = conv_act_bn_dropout_block(merge5, num_filters*16, use_deform=use_deform)
    # conv5 = conv_act_bn_dropout_block(conv5, num_filters*16, use_deform=use_deform)
    
    up5_4 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv5), num_filters*8)
    merge4 = concatenate([conv4, up5_4], axis=3)
    conv4 = conv_act_bn_dropout_block(merge4, num_filters*8, use_deform=use_deform)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=use_deform)
    
    up4_3 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv4), num_filters*4)
    merge3 = concatenate([conv3, up4_3], axis=3)
    conv3 = conv_act_bn_dropout_block(merge3, num_filters*4, use_deform=use_deform)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=use_deform)
    
    up3_2 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv3), num_filters*2)
    merge2 = concatenate([conv2, up3_2], axis=3)
    conv2 = conv_act_bn_dropout_block(merge2, num_filters*2, use_deform=use_deform)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=use_deform)
    
    up2_1 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv2), num_filters)
    merge1 = concatenate([conv1, up2_1], axis=3)
    conv1 = conv_act_bn_dropout_block(merge1, num_filters, use_deform=False)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=False)
    
    text_mask = Conv2D(1, 1, activation='sigmoid', name='text_mask')(conv1)
    
    model = Model(input=input, outputs=text_mask)
    
    global IoU_score
    IoU_score = partial(IoU_score, ignore_last_channel=False)
    IoU_score.__name__ = 'IoU_score'
    
    global custom_loss
    custom_loss = partial(custom_loss, class_weights=None, 
                          loss_weights=loss_weights, ignore_last_channel=False)
    custom_loss.__name__ = 'custom_loss'
    
    model.compile(optimizer=Adam(lr=1e-4), 
                  loss=custom_loss, 
                  metrics=['accuracy', IoU_score])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights, by_name=True)
    
    # model.summary()
    
    return model

def isbi_model(pretrained_weights=None, input_size=(None, None, 3), 
               num_filters=32, use_deform=True, channel_wise=False, 
               normal_conv_trainable=True, loss_weights=[1.0, 1.0]):
    
    global Conv
    Conv = partial(Conv, normal_conv_trainable=normal_conv_trainable,
                   use_deform=use_deform, channel_wise=channel_wise)
   
    def conv_act_bn_dropout_block(input, num_filters, use_deform=True,
                                  dropout=0):
        output = Conv(input, num_filters, use_deform=use_deform)
        output = Activation('relu')(output)
        output = BatchNormalization()(output)
        if dropout > 0:
            output = SpatialDropout2D(dropout)(output)
        
        return output
    
    input = Input(input_size)
    
    conv1 = conv_act_bn_dropout_block(input, num_filters, use_deform=use_deform)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=use_deform)
    down1_2 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = conv_act_bn_dropout_block(down1_2, num_filters*2, use_deform=use_deform)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=use_deform)
    down2_3 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = conv_act_bn_dropout_block(down2_3, num_filters*4, use_deform=use_deform)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=use_deform)
    down3_4 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = conv_act_bn_dropout_block(down3_4, num_filters*8, use_deform=use_deform)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=use_deform)
    conv4 = Dropout(0.5)(conv4)
    down4_5 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = conv_act_bn_dropout_block(down4_5, num_filters*16, use_deform=use_deform)
    conv5 = conv_act_bn_dropout_block(conv5, num_filters*16, use_deform=use_deform)
    conv5 = Dropout(0.5)(conv5)
    
    up5_4 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv5), num_filters*8)
    merge4 = concatenate([conv4, up5_4], axis=3)
    conv4 = conv_act_bn_dropout_block(merge4, num_filters*8, use_deform=use_deform)
    conv4 = conv_act_bn_dropout_block(conv4, num_filters*8, use_deform=use_deform)
    
    up4_3 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv4), num_filters*4)
    merge3 = concatenate([conv3, up4_3], axis=3)
    conv3 = conv_act_bn_dropout_block(merge3, num_filters*4, use_deform=use_deform)
    conv3 = conv_act_bn_dropout_block(conv3, num_filters*4, use_deform=use_deform)
    
    up3_2 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv3), num_filters*2)
    merge2 = concatenate([conv2, up3_2], axis=3)
    conv2 = conv_act_bn_dropout_block(merge2, num_filters*2, use_deform=use_deform)
    conv2 = conv_act_bn_dropout_block(conv2, num_filters*2, use_deform=use_deform)
    
    up2_1 = conv_act_bn_dropout_block(UpSampling2D(size = (2,2))(conv2), num_filters)
    merge1 = concatenate([conv1, up2_1], axis=3)
    conv1 = conv_act_bn_dropout_block(merge1, num_filters, use_deform=False)
    conv1 = conv_act_bn_dropout_block(conv1, num_filters, use_deform=False)
    
    output = Conv2D(1, 1, activation='sigmoid', name='output')(conv1)
    
    model = Model(input=input, outputs=output)
    
    global IoU_score
    IoU_score = partial(IoU_score, ignore_last_channel=False)
    IoU_score.__name__ = 'IoU_score'
    
    global custom_loss
    custom_loss = partial(custom_loss, class_weights=None, 
                          loss_weights=loss_weights, ignore_last_channel=False)
    custom_loss.__name__ = 'custom_loss'
    
    model.compile(optimizer=Adam(lr=1e-4), 
                  loss=custom_loss, 
                  metrics=['accuracy', IoU_score])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights, by_name=True)
    
    # model.summary()
    
    return model

# import random
# h = 1024
# w = 768
# n = 3
# c = 3
# model = text_classification_model(input_size=(h, w, 3), num_filters=1, use_deform=True, ignore_background=True)
# 
# X = np.random.rand(n, h, w, 3)
# y = (np.random.rand(n, h, w, 3) > 0.5) * 1.0
# 
# model.fit(X, y, epochs=1)