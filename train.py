import os
from pathlib import Path

import click
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *

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
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # ckpt_path = os.path.join(checkpoint_dir, 'ep{epoch:03d}_key-iou{val_key_mask_IoU_score:.4f}_value-iou{val_value_mask_IoU_score:.4f}.h5')
    ckpt_path = os.path.join(checkpoint_dir, 'ep{epoch:03d}_loss{val_loss:.4f}_iou{val_IoU_score:.4f}.h5')
    checkpoint = ModelCheckpoint(ckpt_path, 
                                 monitor='val_loss', 
                                 save_weights_only=True, 
                                 save_best_only=True, 
                                 verbose=1)

    global model
    model = Unet(pretrained_weights, input_size=(None, None, 1), num_filters=4, 
                 use_deform=use_deform, channel_wise=channel_wise, 
                 normal_conv_trainable=normal_conv_trainable)
    
    model.compile(optimizer=Adam(lr=1e-4), 
                  loss=custom_categorical_loss, 
                  metrics=['accuracy', IoU_score])

    model.fit_generator(data_generator('dataset/training_data', 2/3, shuffle=True), 
                        steps_per_epoch=99, 
                        validation_data=data_generator('dataset/training_data', -1/3),
                        validation_steps=50,
                        epochs=epochs,
                        callbacks=[checkpoint])

    print(model.evaluate_generator(data_generator('dataset/testing_data'), steps=50))
    
if __name__ == '__main__':
    train()