import os
from pathlib import Path

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import click
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping

from metrics import *
from deform_unet import Unet
from load_data import data_generator

model = None

@click.command()
@click.option('--pretrained_weights', '-w', default=None)
@click.option('--checkpoint_dir', '-s', default='checkpoint/')
@click.option('--deform/--no-deform', '-D/-nD', 'use_deform', default=True)
@click.option('--train-norm-conv/--no-train-norm-conv', '-F/-nF', 'normal_conv_trainable', default=True)
@click.option('--deform-channel/--no-deform-channel', '-C/-nC', 'channel_wise', default=False)
@click.option('--epochs', '-e', default=100)
def train(pretrained_weights, checkpoint_dir, use_deform, channel_wise,
          normal_conv_trainable, epochs):
    
# pretrained_weights=None
# checkpoint_dir='checkpoint/'
# use_deform=False
# normal_conv_trainable=True
# channel_wise=False
# epochs=1
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # ckpt_path = os.path.join(checkpoint_dir, 'ep{epoch:03d}_key-iou{val_key_mask_IoU_score:.4f}_value-iou{val_value_mask_IoU_score:.4f}.h5')
    ckpt_path = os.path.join(checkpoint_dir, 'ep{epoch:03d}_loss{val_loss:.4f}_iou{val_IoU_score:.4f}.h5')
    cp = ModelCheckpoint(ckpt_path, monitor='val_loss',  save_weights_only=True, 
                         save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', min_delta=0, restore_best_weights=True)

    model_args = dict(input_size=(None, None, 1), num_classes=3, num_filters=4, 
                      use_deform=use_deform, channel_wise=channel_wise, 
                      normal_conv_trainable=normal_conv_trainable)

    # global model
    model = Unet(pretrained_weights, **model_args)
    model.summary()

    model.fit_generator(data_generator('dataset/training_data', 2/3, shuffle=True), 
                        steps_per_epoch=99, 
                        validation_data=data_generator('dataset/training_data', -1/3),
                        validation_steps=50,
                        epochs=epochs,
                        callbacks=[cp, es])

    val_result = model.evaluate_generator(data_generator('dataset/training_data', -1/3), steps=50)
    test_result = model.evaluate_generator(data_generator('dataset/testing_data'), steps=50)
    print(val_result)
    print(test_result)

    save_path = '_'.join(['mask2mask', str(model_args['num_classes']) + 'C',
                          'D' if use_deform else ('D_C' if channel_wise else 'D_nC'),
                          'val{:.4f}'.format(val_result[-1]),
                          'test{:.4f}'.format(test_result[-1])]) + '.h5'
    model.save(save_path)
    model.load_weights(save_path)
    print(model.evaluate_generator(data_generator('dataset/training_data', -1/3), steps=50))
    print(model.evaluate_generator(data_generator('dataset/testing_data'), steps=50))

    del model
    model = Unet(None, **model_args)
    model.load_weights(save_path)
    print(model.evaluate_generator(data_generator('dataset/training_data', -1/3), steps=50))
    print(model.evaluate_generator(data_generator('dataset/testing_data'), steps=50))
    
if __name__ == '__main__':
    train()