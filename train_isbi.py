import os
from pathlib import Path
from functools import partial

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import click
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from metrics import *
from deform_unet import isbi_model
from load_isbi import *

model = None

@click.command()
@click.option('--pretrained_weights', '-w', default=None)
@click.option('--epochs', '-e', default=100)
@click.option('--checkpoint_dir', '-s', default='checkpoint/')
@click.option('--deform/--no-deform', '-D/-nD', 'use_deform', default=True)
@click.option('--channel-wise-deform', '-C/-nC', default=False)
@click.option('--normal-conv-trainable', '-N/-nN', default=True)
def train(pretrained_weights, epochs, checkpoint_dir, use_deform, 
          channel_wise_deform, normal_conv_trainable):
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # ckpt_path = os.path.join(checkpoint_dir, 'ep{epoch:03d}_key-iou{val_key_mask_IoU_score:.4f}_value-iou{val_value_mask_IoU_score:.4f}.h5')
    ckpt_path = os.path.join(checkpoint_dir, 'ep{epoch:03d}_loss{val_loss:.4f}_iou{val_IoU_score:.4f}.h5')
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_loss',  save_weights_only=False, 
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=16,
                      restore_best_weights=True, verbose=1)
    ]

    model_args = dict(input_size=(None, None, 1), num_filters=64, 
                      use_deform=use_deform, channel_wise=channel_wise_deform, 
                      normal_conv_trainable=normal_conv_trainable,
                      loss_weights=[1.0, 0.5])

    # global model
    model = isbi_model(pretrained_weights, **model_args)
    model.summary()

    model.fit_generator(zip(train_image_gen, train_mask_gen), 
                        steps_per_epoch=50, 
                        validation_data=zip(val_image_gen, val_mask_gen),
                        validation_steps=1,
                        epochs=epochs,
                        callbacks=callbacks)

    train_image_gen_no_aug = \
        ImageDataGenerator(rescale=1./255)\
        .flow_from_directory('ISBI/train', classes=['image'], target_size=(512, 512),
                             color_mode='grayscale', class_mode=None, batch_size=20)
    train_mask_gen_no_aug = \
        ImageDataGenerator(rescale=1./255)\
        .flow_from_directory('ISBI/train', classes=['label'], target_size=(512, 512),
                             color_mode='grayscale', class_mode=None, batch_size=20)
    
    train_result = model.evaluate_generator(zip(train_image_gen_no_aug, train_mask_gen_no_aug), 
                                            steps=1)
    
    val_result = model.evaluate_generator(zip(val_image_gen, val_mask_gen), 
                                          steps=1)
    
    test_image_gen = \
        ImageDataGenerator(rescale=1./255)\
        .flow_from_directory('ISBI/my_test', classes=['image'], target_size=(512, 512),
                             color_mode='grayscale', class_mode=None, batch_size=5)
    test_mask_gen = \
        ImageDataGenerator(rescale=1./255)\
        .flow_from_directory('ISBI/my_test', classes=['label'], target_size=(512, 512),
                             color_mode='grayscale', class_mode=None, batch_size=5)
    
    test_result = model.evaluate_generator(zip(test_image_gen, test_mask_gen), 
                                          steps=1)
    print(train_result)
    print(val_result)
    print(test_result)

    save_path = '_'.join(['ISBI',
                          'nD' if not use_deform else ('D_C' if channel_wise_deform else 'D_nC'),
                          'train{:.4f}'.format(train_result[-1]),
                          'val{:.4f}'.format(val_result[-1]),
                          'test{:.4f}'.format(test_result[-1])]) + '.h5'
    model.save(save_path)
    
if __name__ == '__main__':
    train()