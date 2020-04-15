import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

def weighted_binary_crossentropy(y_true, y_pred, class_weights=None):

    # Original binary crossentropy (see losses.py):
    # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    # Calculate the binary crossentropy
    b_ce = K.binary_crossentropy(y_true, y_pred)
    
    if class_weights is None:
        return b_ce
    else:
        class_weights = np.array(class_weights) / np.sum(class_weights)
    
    y_shape = list(K.int_shape(y_pred))
    if len(class_weights) != y_shape[-1]:
        raise ValueError('''Number of weights ({}) does not match number of 
                         classes ({})'''.format(len(class_weights), y_shape[-1]))

    # Apply the weights
    weight_vector = y_true * class_weights[1] + (1. - y_true) * class_weights[0]
    weighted_b_ce = weight_vector * b_ce

    # Return the mean error
    return K.mean(weighted_b_ce)


def weighted_categorical_crossentropy(y_true, y_pred, class_weights=None):

    # Original binary crossentropy (see losses.py):
    # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    # Calculate the crossentropy
    loss_map = K.categorical_crossentropy(y_true, y_pred)
    
    if class_weights is None:
        return loss_map
    else:
        class_weights = np.array(class_weights) / np.sum(class_weights)

    y_shape = list(K.int_shape(y_pred))
    if len(class_weights) != y_shape[-1]:
        raise ValueError('''Number of weights ({}) does not match number of 
                         classes ({})'''.format(len(class_weights), y_shape[-1]))
    
    # Compute the weight
    weight_map = K.zeros_like(y_pred[..., 0])
    for i in range(len(class_weights)):
        weight_map += y_true[..., i] * class_weights[i]
    
    # Apply the weights
    loss_map = loss_map * weight_map

    # Return the mean error
    return K.mean(loss_map)


def dice_loss(y_true, y_pred, ignore_last_channel, smooth=1e-6):
    
    if ignore_last_channel:
        y_true = y_true[..., :-1]
        y_pred = y_pred[..., :-1]
    
    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])
    # print('-' * 50)
    # print('dice_coef')
    # print(K.int_shape(y_true), y_true.shape.as_list())
    # print(K.int_shape(intersection), intersection.shape.as_list())
    # print(K.int_shape(union), union.shape.as_list())
    # print('-' * 50)
    loss = K.mean((2. * intersection + smooth) / (union + smooth), axis=-1)
    loss = K.mean(loss)
    
    return -loss


def custom_loss(y_true, y_pred, class_weights=[0.1, 0.9], 
                loss_weights=[4, 0.5], ignore_last_channel=False):
    
    dice = dice_loss(y_true, y_pred, ignore_last_channel=ignore_last_channel)
    cross_entropy = weighted_binary_crossentropy(y_true, y_pred, class_weights)
    
    return loss_weights[0] * dice + loss_weights[1] * cross_entropy


def custom_categorical_loss(y_true, y_pred, class_weights=[1, 1, 1, 0.3], 
                            loss_weights=[4, 0.5], ignore_last_channel=False):
    
    dice = dice_loss(y_true, y_pred, ignore_last_channel=ignore_last_channel)
    
    cross_entropy = weighted_categorical_crossentropy(y_true, y_pred, class_weights)
    
    return loss_weights[0] * dice + loss_weights[1] * cross_entropy


def IoU_score(y_true, y_pred, smooth=1e-6, ignore_last_channel=False):
    
    if ignore_last_channel:
        y_true = y_true[..., :-1]
        y_pred = y_pred[..., :-1]
    
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