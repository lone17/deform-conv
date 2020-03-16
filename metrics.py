import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

def weighted_binary_crossentropy(class_weights):

    def loss(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * class_weights[1] + (1. - y_true) * class_weights[0]
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return loss

def weighted_categorical_crossentropy(weights):

    def loss(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the crossentropy
        loss_map = K.categorical_crossentropy(y_true, y_pred)

        y_shape = list(K.int_shape(y_pred))
        if len(weights) != y_shape[-1]:
            raise ValueError('''Number of weights ({}) does not match number of 
                             classes ({})'''.format(len(weights), y_shape[-1]))
        
        # Compute the weight
        weight_map = K.zeros_like(y_pred[..., 0])
        print(y_pred.shape.as_list())
        print(loss_map.shape.as_list())
        print(weight_map.shape.as_list())
        for i in range(len(weights)):
            weight_map += y_true[..., i] * weights[i]
        
        # Apply the weights
        loss_map = loss_map * weight_map

        # Return the mean error
        return K.mean(loss_map)

    return loss

def dice_coef(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    # print('-' * 50)
    # print('dice_coef')
    # print(K.int_shape(y_true), y_true.shape.as_list())
    # print(K.int_shape(intersection), intersection.shape.as_list())
    # print(K.int_shape(union), union.shape.as_list())
    # print('-' * 50)
    loss = K.mean( (2. * intersection + smooth) / (union + smooth), axis=-1)
    loss = K.mean(loss)
    
    return loss

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def custom_loss(y_true, y_pred, class_weights=[0.1, 0.9]):
    dice = dice_loss(y_true, y_pred)
    if class_weights is not None:
        cross_entropy = weighted_binary_crossentropy(class_weights)(y_true, y_pred)
    else:
        cross_entropy = K.binary_crossentropy(y_true, y_pred)
    return 4 * dice + 0.5 * cross_entropy

def custom_categorical_loss(y_true, y_pred, class_weights=[1, 1, 1, 0.3]):
    class_weights = np.array(class_weights) / np.sum(class_weights)
    dice = dice_loss(y_true, y_pred)
    if class_weights is not None:
        cross_entropy = weighted_categorical_crossentropy(class_weights)(y_true, y_pred)
    else:
        cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    return 4 * dice + 0.5 * cross_entropy

def IoU_score(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
    union = K.sum(y_true, [1, 2]) + K.sum(y_pred, [1, 2]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=-1)
    iou = K.mean(iou)
    # print('-' * 50)
    # print('IoU_score')
    # print(K.int_shape(y_true), y_true.shape.as_list())
    # print(K.int_shape(intersection), intersection.shape.as_list())
    # print(K.int_shape(union), union.shape.as_list())
    # print('-' * 50)
    return iou

custom_objects = [IoU_score, custom_loss, custom_categorical_loss]