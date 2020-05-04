import os
import random
from pathlib import Path
from functools import partial

import cv2
from imutils import paths
from keras.models import load_model
from matplotlib import pyplot as plt

from utils import *
from metrics import *
from load_data import *
from deform_unet import *
from deform_conv.layers import ConvOffset2D

def calc_mIoU(a, b):
    i = (a * b).sum(axis=(1,2))
    print(a.shape, b.shape)
    u = a.sum(axis=(1,2)) + b.sum(axis=(1,2)) - i
    iou = np.mean((i + 1e-6) / (u + 1e-6), axis=-1)
    iou = np.mean(iou)
    
    return iou

IoU_score = partial(IoU_score, ignore_last_channel=True)
IoU_score.__name__ = 'IoU_score'

custom_categorical_loss = partial(custom_categorical_loss, 
                                  class_weights=[1, 1, 1, 0.3],
                                  loss_weights=[4.0, 0.5],
                                  ignore_last_channel=True)
custom_categorical_loss.__name__ = 'custom_categorical_loss'

# ConvOffset2D = partial(ConvOffset2D, channel_wise=True)
# ConvOffset2D.__name__ = 'ConvOffset2D'
    
custom_objects = {
    'IoU_score': IoU_score,
    'custom_categorical_loss': custom_categorical_loss, 
    'custom_loss': custom_loss, 
    'ConvOffset2D': ConvOffset2D
}

weight_path = '1c_mask2mask4C_nD_nI_val0.7241_test0.5006.h5'
# model = Unet_text(pretrained_weights=weight_path, input_size=(None, None, 2), 
#                   num_filters=16, num_classes=4, use_deform=False, channel_wise=False,
#                   class_weights=[1, 1, 1, 0.3], loss_weights=[4.0, 0.5],
#                   ignore_background=False)
model = load_model(weight_path, custom_objects=custom_objects)

# print(model.evaluate_generator(data_generator('dataset/testing_data'), steps=50))

img_paths = paths.list_images('dataset/training_data/images')
img_paths = list(img_paths)
img_paths = sorted(img_paths, key=lambda x: get_file_name(x))
img_paths = img_paths[:round(len(img_paths) * 2/3)]
# random.shuffle(img_paths)

down_scale = 32

miou = []
for p in img_paths[:]:
    if '00851772_1780' not in p:
        continue
    # print(p)
    
    # load label
    annotation_path = os.path.splitext(p)[0].replace('images', 'adjusted_annotations') + '.json'
    annotations = read_json(annotation_path)['form']
    
    # load image
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    
    # convert image and label to input and output masks
    resized_grey_image, all_text_mask, *gt_masks = \
        image_to_text_masks(img, annotations, down_scale)
    
    # make input
    input = np.dstack([resized_grey_image, all_text_mask])
    
    # # padding input to square
    # canvas = np.zeros((max(input.shape), max(input.shape), input.shape[-1]))
    # canvas[:input.shape[0], :input.shape[1]] = input
    # input = canvas
    
    # make ground-truth
    gt_masks = np.dstack(gt_masks)
    
    # get model outputs
    outputs = model.predict(input[None, ...])
    
    # # discard padded sections from the outputs
    # processed_shape = resized_grey_image.shape
    # outputs = outputs[:, :processed_shape[0], :processed_shape[1]]
    
    # plot outputs
    plt.subplot('221')
    plt.title('key mask')
    plt.imshow(outputs[0][..., 0])
    
    plt.subplot('222')
    plt.title('value mask')
    plt.imshow(outputs[0][..., 1])
    
    plt.subplot('223')
    plt.title('other mask')
    plt.imshow(outputs[0][..., 2])
    
    plt.subplot('224')
    plt.title('background mask')
    plt.imshow(outputs[0][..., 3])
    plt.show()
    
    # rprocess image for drawing overlay
    img = cv2.resize(img, processed_shape[:2][::-1]) / 255
    
    # convert model outputs to a onehot matrix
    onehot_mask = (outputs[0].max(axis=-1)[..., None] == outputs[0]) * 1.0
    
    # draw the onehot maxtrix overlaying the input image
    pred_overlay_img = np.zeros((processed_shape[0], processed_shape[1], 3))
    pred_overlay_img[onehot_mask[..., 0] > 0] = [0, 0, 1] # key
    pred_overlay_img[onehot_mask[..., 1] > 0] = [0, 1, 0] # value
    pred_overlay_img[onehot_mask[..., 2] > 0] = [1, 0, 0] # other
    pred_overlay_img = cv2.addWeighted(img, 0.7, pred_overlay_img, 0.3, 0)
    
    # draw the pre-onehot outputs overlay the input images
    # pred_overlay_img2 = cv2.addWeighted(img.astype('float32'), 0.7, outputs[0][..., :3], 0.3, 0)
    
    # draw the ground-truth overlaying the input image
    gt_overlay_img = np.zeros((processed_shape[0], processed_shape[1], 3))
    gt_overlay_img[gt_masks[..., 0] > 0] = [0, 0, 1] # key
    gt_overlay_img[gt_masks[..., 1] > 0] = [0, 1, 0] # value
    gt_overlay_img[gt_masks[..., 2] > 0] = [1, 0, 0] # other
    gt_overlay_img = cv2.addWeighted(img, 0.7, gt_overlay_img, 0.3, 0)
    
    # plot overlays
    plt.subplot('121')
    plt.imshow(gt_overlay_img)
    plt.subplot('122')
    plt.imshow(pred_overlay_img)
    plt.show()
    
    # save debug image
    save_path = Path(os.path.splitext(weight_path)[0] + '_training\\')
    save_path.mkdir(exist_ok=True)
    # cv2.imwrite(str(save_path.joinpath(Path(p).name).with_suffix('.png')), 
    #             cv2.cvtColor((pred_overlay_img * 255).astype('uint8'), cv2.COLOR_BGR2RGB))
    
    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.imshow(outputs[0][..., i])
    # plt.show()
    
    # calculate mean IoU of the current image
    miou.append(calc_mIoU(gt_masks[None, ...], outputs))

# the mIoU of the dataset
print(np.mean(miou))