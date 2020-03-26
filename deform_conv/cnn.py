from __future__ import absolute_import, division


from keras.layers import *
from deform_conv.layers import ConvOffset2D


def get_cnn():
    inputs = l = Input((None, None, 1), name='input')

    # conv11
    l = Conv2D(32, (3, 3), padding='same', name='conv11')(l)
    l = Activation('relu', name='conv11_relu')(l)
    l = BatchNormalization(name='conv11_bn')(l)

    # conv12
    l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12')(l)
    l = Activation('relu', name='conv12_relu')(l)
    l = BatchNormalization(name='conv12_bn')(l)

    # conv21
    l = Conv2D(128, (3, 3), padding='same', name='conv21')(l)
    l = Activation('relu', name='conv21_relu')(l)
    l = BatchNormalization(name='conv21_bn')(l)

    # conv22
    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv22')(l)
    l = Activation('relu', name='conv22_relu')(l)
    l = BatchNormalization(name='conv22_bn')(l)

    # out
    l = GlobalAvgPool2D(name='avg_pool')(l)
    l = Dense(10, name='fc1')(l)
    outputs = l = Activation('softmax', name='out')(l)

    return inputs, outputs


def get_deform_cnn(trainable, channel_wise=True):
    inputs = l = Input((None, None, 1), name='input')

    # conv11
    l = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv11', 
               trainable=trainable)(l)
    l = BatchNormalization(name='conv11_bn')(l)

    # conv12
    l_offset = ConvOffset2D(32, channel_wise=channel_wise, name='conv12_offset')(l)
    l = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv12', 
               trainable=trainable)(l_offset)
    l = MaxPooling2D((2, 2))(l)
    l = BatchNormalization(name='conv12_bn')(l)

    # conv21
    l_offset = ConvOffset2D(64, channel_wise=channel_wise, name='conv21_offset')(l)
    l = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv21', 
               trainable=trainable)(l_offset)
    l = BatchNormalization(name='conv21_bn')(l)

    # conv22
    l_offset = ConvOffset2D(128, channel_wise=channel_wise, name='conv22_offset')(l)
    l = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv22', 
               trainable=trainable)(l_offset)
    l = MaxPooling2D((2, 2))(l)
    l = BatchNormalization(name='conv22_bn')(l)

    # out
    l = GlobalAvgPool2D(name='avg_pool')(l)
    outputs = Dense(10, activation='softmax', name='fc1', 
                    trainable=trainable)(l)

    return inputs, outputs

