import random
from pathlib import Path

import cv2
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt

from utils import *

def preprocess_image(image):
    if np.max(image) > 1:
        image = image / 255
    
    img_h, img_w = image.shape[:2]
    img_h = round_up_dividend(img_h, 16)
    img_w = round_up_dividend(img_w, 16)
    image = cv2.resize(image, (img_w, img_h))
    
    return image

def image_to_masks(image, annotations):
    img_h, img_w = image.shape[:2]
    img_h = round_up_dividend(img_h, 16)
    img_w = round_up_dividend(img_w, 16)
    ratio_h = img_h / image.shape[0]
    ratio_w = img_w / image.shape[1]
    image = cv2.resize(image, (img_w, img_h))
    
    all_text_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    key_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    value_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    other_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    # annotations = sorted(annotations, key=lambda x: x['id'])
    for field in annotations:
        x1, y1, x2, y2 = (np.array(field['box']) * [ratio_w, ratio_h, ratio_w, ratio_h]).astype(int)
        all_text_mask[y1:y2, x1:x2] = 1.0
        if field['label'] == 'question':
            key_mask[y1:y2, x1:x2] = 1.0
        elif field['label'] == 'answer':
            value_mask[y1:y2, x1:x2] = 1.0
        else:
            other_mask[y1:y2, x1:x2] = 1.0
    background = (1 - key_mask) * (1 - value_mask) * (1 - other_mask)
    
    # plt.subplot('231')
    # plt.imshow(image)
    # plt.subplot('232')
    # plt.imshow(all_text_mask[..., 0])
    # plt.subplot('233')
    # plt.imshow(key_mask[..., 0])
    # plt.subplot('234')
    # plt.imshow(value_mask[..., 0])
    # plt.subplot('235')
    # plt.imshow(other_mask[..., 0])
    # plt.subplot('236')
    # plt.imshow(background[..., 0])
    # plt.show()
    
    
    return all_text_mask, key_mask, value_mask, other_mask, background

def data_generator(data_dir, portion=1.0, shuffle=False):
    image_map = {get_file_name(p): p for p in paths.list_images(Path(data_dir) / 'images')}
    label_map = {get_file_name(p): p for p in paths.list_files(Path(data_dir) / 'annotations', validExts=('.json'))}
    
    chosen_keys = sorted(list(image_map.keys()))
    amount = round(len(chosen_keys) * portion)
    chosen_keys = chosen_keys[:amount]

    while True:
        if shuffle:
            random.shuffle(chosen_keys)
        
        for k in chosen_keys:
            # image = cv2.imread(image_map[k], cv2.IMREAD_GRAYSCALE) / 255
            # image = 1.0 - image
            image = cv2.imread(image_map[k], cv2.IMREAD_COLOR) / 255
            annotations = read_json(label_map[k])['form']
            
            all_text_mask, *output_masks = image_to_masks(image, annotations)
            
            output_masks = np.dstack(output_masks)
            # plt.subplot('121')
            # plt.imshow(output_masks[..., :-1] * 255)
            # plt.subplot('122')
            # plt.imshow(output_masks[..., -1])
            # plt.show()
            yield (all_text_mask[None, ...], output_masks[None, ...])

if __name__ == '__main__':
    data_generator('dataset//training_data')