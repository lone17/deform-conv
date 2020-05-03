import random
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt

from utils import *

def resize_to_fit_downscale(image, down_scale=16):
    img_h, img_w = image.shape[:2]
    img_h = round_up_dividend(img_h, down_scale)
    img_w = round_up_dividend(img_w, down_scale)
    image = cv2.resize(image, (img_w, img_h))
    
    return image

def image_to_text_masks(image, annotations, down_scale):
    original_img_h, original_img_w = image.shape[:2]
    image = resize_to_fit_downscale(image, down_scale)
    img_h, img_w = image.shape[:2]
    ratio_h = img_h / original_img_h
    ratio_w = img_w / original_img_w
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = 1.0 - image / 255
    
    all_text_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    key_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    value_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    other_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    
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
    # background = (1 - key_mask) * (1 - value_mask)
    
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
    
    
    return (image,
            all_text_mask, 
            key_mask, 
            value_mask, 
            other_mask, 
            background)

def image_to_relation_masks(image, annotations, down_scale):
    original_img_h, original_img_w = image.shape[:2]
    image = resize_to_fit_downscale(image, down_scale)
    img_h, img_w = image.shape[:2]
    ratio_h = img_h / original_img_h
    ratio_w = img_w / original_img_w
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = 1.0 - image / 255
    
    all_text_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    key_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    value_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    other_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
    
    annotations_map = {str(x['id']): x for x in annotations}
    for field in annotations:
        for word in field['words']:
            x1, y1, x2, y2 = (np.array(word['box']) * [ratio_w, ratio_h, ratio_w, ratio_h]).astype(int)
            all_text_mask[y1:y2, x1:x2] = 1.0
    
    horizontal_relation_mask, vertical_relation_mask = \
        draw_relation(all_text_mask.shape, annotations_map, colour=1)
    
    # plt.subplot('231')
    # plt.imshow(image)
    # 
    # plt.subplot('232')
    # blended = (1 - image) * 0.6 + (1 - all_text_mask[..., 0]) * 0.4
    # plt.imshow(blended)
    # 
    # blended = (1 - image) * 0.6 + (1 - horizontal_relation_mask[..., 0]) * 0.4
    # plt.subplot('234')
    # plt.imshow(blended)
    # 
    # blended = (1 - image) * 0.6 + (1 - vertical_relation_mask[..., 0]) * 0.4
    # plt.subplot('235')
    # plt.imshow(blended)
    # 
    # relation_mask = np.maximum.reduce([all_text_mask, 
    #                                    horizontal_relation_mask, 
    #                                    vertical_relation_mask])
    # blended = (1 - image) * 0.6 + (1 - relation_mask[..., 0]) * 0.4
    # plt.subplot('236')
    # plt.imshow(blended)
    # 
    # plt.show()
    
    
    return (image,
            all_text_mask, 
            horizontal_relation_mask,
            vertical_relation_mask)

@lru_cache(None)
def read_img(img_path):
    return cv2.imread(img_path, cv2.IMREAD_COLOR)

mask_cache = {}
def data_generator(data_dir, mask_type, portion=1.0, down_scale=16, shuffle=False):
    image_path = Path(data_dir) / 'images'
    image_map = {get_file_name(p): p 
                 for p in paths.list_images(image_path)}
    
    annotation_path = Path(data_dir) / 'adjusted_annotations'
    if not annotation_path.exists():
        annotation_path = Path(data_dir) / 'annotations'
    label_map = {get_file_name(p): p 
                 for p in paths.list_files(annotation_path, validExts=('.json'))}
    
    chosen_keys = sorted(list(image_map.keys()))
    amount = round(len(chosen_keys) * portion)
    chosen_keys = chosen_keys[:amount]

    while True:
        if shuffle:
            random.shuffle(chosen_keys)
        
        for k in chosen_keys:
            # image = cv2.imread(image_map[k], cv2.IMREAD_GRAYSCALE) / 255
            # image = 1.0 - image
            image = read_img(image_map[k])
            annotations = read_json(label_map[k])['form']
            
            if mask_type == 'text':
                if k not in mask_cache:
                    mask_cache[k] = image_to_text_masks(image, annotations, 
                                                        down_scale)
                resized_grey_image, all_text_mask, *output_masks = mask_cache[k]
            elif mask_type == 'relation':
                if k not in mask_cache:
                    mask_cache[k] = image_to_relation_masks(image, annotations, 
                                                            down_scale)
                resized_grey_image, all_text_mask, *output_masks = mask_cache[k]
            
            input = np.dstack([resized_grey_image, all_text_mask])
            output_masks = np.dstack(output_masks)
            # plt.subplot('121')
            # plt.imshow(output_masks[..., 0] * 255)
            # plt.subplot('122')
            # plt.imshow(output_masks[..., 1])
            # plt.show()
            yield (input[None, ...], output_masks[None, ...])

if __name__ == '__main__':
    data_generator('dataset//training_data', mask_type='relation')
    # sizes = []
    # max_w, max_h = 0, 0
    # for p in paths.list_images('dataset//training_data//images'):
    #     img = cv2.imread(p)
    #     h, w = img.shape[:2]
    #     sizes.append((w, h))
    #     if h > max_h:
    #         max_h = h
    #     if w > max_w:
    #         max_w = w
    # 
    # img = np.zeros((max_h, max_w))
    # for w, h in sizes:
    #     img[:h, :w] += 1
    # 
    # img /= np.max(img)
    # 
    # plt.imshow(img)
    # plt.show()
    # 
    # plt.subplot('121')
    # plt.hist([x[0] for x in sizes])
    # plt.subplot('122')
    # plt.hist([x[1] for x in sizes])
    # plt.show()
