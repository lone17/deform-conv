from keras.preprocessing.image import ImageDataGenerator

data_gen_args = dict(rescale=1./255,
                     rotation_range=0.2,
                     shear_range=0.2,
                     zoom_range=0.2,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=True,
                     fill_mode='nearest')

seed = 17
train_image_gen = \
    ImageDataGenerator(**data_gen_args)\
    .flow_from_directory('ISBI/train', classes=['image'], target_size=(512, 512),
                         color_mode='grayscale', class_mode=None, batch_size=1, 
                         shuffle=True, seed=seed)
train_mask_gen = \
    ImageDataGenerator(**data_gen_args)\
    .flow_from_directory('ISBI/train', classes=['label'], target_size=(512, 512),
                         color_mode='grayscale', class_mode=None, batch_size=1, 
                         shuffle=True, seed=seed)

val_image_gen = \
    ImageDataGenerator(rescale=1./255)\
    .flow_from_directory('ISBI/val', classes=['image'], target_size=(512, 512),
                         color_mode='grayscale', class_mode=None, batch_size=1)
val_mask_gen = \
    ImageDataGenerator(rescale=1./255)\
    .flow_from_directory('ISBI/val', classes=['label'], target_size=(512, 512),
                         color_mode='grayscale', class_mode=None, batch_size=1)

