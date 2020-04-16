import os
from pathlib import Path
from functools import partial

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import click
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from metrics import *
from deform_unet import Unet
from load_data import data_generator

data_generator = partial(data_generator, down_scale=32)

model = None

@click.command()
@click.option('--pretrained_weights', '-w', default=None)
@click.option('--epochs', '-e', default=1000)
@click.option('--checkpoint_dir', '-s', default='checkpoint/')
@click.option('--deform/--no-deform', '-D/-nD', 'use_deform', default=True)
@click.option('--channel-wise-deform', '-C/-nC', default=False)
@click.option('--normal-conv-trainable', '-N/-nN', default=True)
@click.option('--ignore-background', '-I/-nI', default=False)
def train(pretrained_weights, epochs, checkpoint_dir, use_deform, 
          channel_wise_deform, normal_conv_trainable, ignore_background):
# pretrained_weights=None
# checkpoint_dir='checkpoint/'
# use_deform=False
# normal_conv_trainable=True
# channel_wise_deform=False
# ignore_background=False
# epochs=1
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # ckpt_path = os.path.join(checkpoint_dir, 'ep{epoch:03d}_key-iou{val_key_mask_IoU_score:.4f}_value-iou{val_value_mask_IoU_score:.4f}.h5')
    ckpt_path = os.path.join(checkpoint_dir, '1a_ep{epoch:03d}_loss{val_loss:.4f}_iou{val_IoU_score:.4f}.h5')
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_loss',  save_weights_only=False, 
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=50,
                      restore_best_weights=True, verbose=1)
    ]

    model_args = dict(input_size=(None, None, 2), num_classes=4, num_filters=4, 
                      use_deform=use_deform, channel_wise=channel_wise_deform, 
                      normal_conv_trainable=normal_conv_trainable,
                      class_weights=[1, 1, 1, 0.3], loss_weights=[4.0, 0.5],
                      ignore_background=ignore_background)

    # global model
    model = Unet(pretrained_weights, **model_args)
    model.summary()

    model.fit_generator(data_generator('dataset/training_data', 2/3, shuffle=True), 
                        steps_per_epoch=50, 
                        validation_data=data_generator('dataset/training_data', -1/3),
                        validation_steps=50,
                        epochs=epochs,
                        callbacks=callbacks)

    val_result = model.evaluate_generator(data_generator('dataset/training_data', -1/3), steps=50)
    test_result = model.evaluate_generator(data_generator('dataset/testing_data'), steps=50)
    print(val_result)
    print(test_result)

    save_path = '_'.join(['1a_mask2mask' + str(model_args['num_classes']) + 'C',
                          'nD' if not use_deform else ('D_C' if channel_wise_deform else 'D_nC'),
                          'I' if ignore_background else 'nI',
                          'val{:.4f}'.format(val_result[-1]),
                          'test{:.4f}'.format(test_result[-1])]) + '.h5'
    model.save(save_path)
    # model.load_weights(save_path)
    # print(model.evaluate_generator(data_generator('dataset/training_data', -1/3), steps=50))
    # print(model.evaluate_generator(data_generator('dataset/testing_data'), steps=50))
    # 
    # del model
    # model = Unet(save_path, **model_args)
    # print(model.evaluate_generator(data_generator('dataset/training_data', -1/3), steps=50))
    # print(model.evaluate_generator(data_generator('dataset/testing_data'), steps=50))
    
if __name__ == '__main__':
    train()