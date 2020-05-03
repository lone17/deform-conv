import os
import json
import math
import traceback as tb
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
from imutils import paths
from matplotlib import pyplot as plt

grey = (150, 150, 150)
blue = (0, 0, 255)
green = (0, 255, 0)
yellow = (255, 255, 0)
magenta = (255, 0, 255)

def get_file_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def read_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    
    return content

def round_up_dividend(num, divisor):
    return num - (num % divisor) + (divisor * num % divisor)

def round(num):
    num = math.ceil(num) if num - math.floor(num) > 0.5 else math.floor(num) 
    
    return num

def draw_link(image, pt1, pt2, colour=255, thinkness=6):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    center = (pt1 / 2 + pt2 / 2).astype(int)
    one_fourth = (pt1 / 2 + center / 2).astype(int)
    three_fourth = (center / 2 + pt2 / 2).astype(int)
    height = thinkness
    width = int(np.linalg.norm(pt1 - pt2))
    if pt1[0] - pt2[0] != 0:
        angle = np.degrees(np.arctan((pt1[1] - pt2[1]) / (pt1[0] - pt2[0])))
    else:
        angle = 90
    
    cv2.ellipse(image, tuple(center), (width // 2, height // 2), angle, 0, 360, 
                colour, cv2.FILLED)
    cv2.ellipse(image, tuple(one_fourth), (width // 4, height // 2), angle, 0, 360, 
                colour, cv2.FILLED)
    cv2.ellipse(image, tuple(three_fourth), (width // 4, height // 2), angle, 0, 360, 
                colour, cv2.FILLED)
    
    return image


def draw_relation(img_shape, annotations_map, colour=255):
    horizontal_relation_mask = np.zeros(img_shape, dtype=np.uint8)
    vertical_relation_mask = np.zeros(img_shape, dtype=np.uint8)
    for line1_id, line1 in annotations_map.items():
        line1_left, line1_top, line1_right, line1_bottom = line1['box']
        if line1['label'] in ['header', 'other']:
            continue
        if line1['label'] in ['question', 'answer']:
            for line2_id in get_children(line1_id, annotations_map):
                line2 = annotations_map[line2_id]
                line2_left, line2_top, line2_right, line2_bottom = line2['box']
                for word1 in line1['words']:
                    for word2 in line2['words']:
                        left1, top1, right1, bottom1 = np.array(word1['box'])
                        left2, top2, right2, bottom2 = np.array(word2['box'])
                        center1 = (int((left1 + right1) / 2), int((top1 + bottom1) / 2))
                        center2 = (int((left2 + right2) / 2), int((top2 + bottom2) / 2))
                        # cv2.line(image, center1, center2, magenta, thickness=3)
                        if (   max(line1_top, line2_top) < min(line1_bottom, line2_bottom)
                            or (    line2_left > line1_right 
                                and line2_right > line1_right + (line1_right - line1_left) * 2)):
                            draw_link(horizontal_relation_mask, center1, center2, 
                                      colour, thinkness=8)
                        else:
                            draw_link(vertical_relation_mask, center1, center2, 
                                      colour, thinkness=8)
                        # draw_link(image, (left1, top1), (left2, top2), colour=magenta, thinkness=2)
                        # draw_link(image, (left1, bottom1), (left2, bottom2), colour=magenta, thinkness=2)
                        # draw_link(image, (right1, top1), (right2, top2), colour=magenta, thinkness=2)
                        # draw_link(image, (right1, bottom1), (right2, bottom2), colour=magenta, thinkness=2)
        else:
            print(line1['label'])
    
    return horizontal_relation_mask, vertical_relation_mask

def get_children(id, annotations):
    children = []
    id = str(id)
    for l in annotations[str(id)]['linking']:
        line1, line2 = [str(i) for i in l]
        if id == line1:
            children.append(line2)
            if annotations[line2]['label'] in ['question', 'answer']:
                children += get_children(line2, annotations)
    return children

def visualize_data(image_dir, label_dir, out_dir):
    
    def visualize(image, annotations):
        
        img_h, img_w = image.shape[:2]
        
        text_mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        annotations_map = {str(x['id']): x for x in annotations}
        links = set()
        checked = set()
        
        for field in annotations_map.values():
            x1, y1, x2, y2 = np.array(field['box'])
            label = field['label']
            if label == 'header':
                text_mask[y1:y2, x1:x2] = np.maximum(text_mask[y1:y2, x1:x2], yellow)
            elif label == 'question':
                text_mask[y1:y2, x1:x2] = np.maximum(text_mask[y1:y2, x1:x2], blue)
            elif label == 'answer':
                text_mask[y1:y2, x1:x2] = np.maximum(text_mask[y1:y2, x1:x2], green)
            elif label == 'other':
                text_mask[y1:y2, x1:x2] = np.maximum(text_mask[y1:y2, x1:x2], grey)
            
            for l in field['linking']:
                # links.add(tuple(l))
                item = tuple(l)
                if item in checked and item not in links:
                    links.add(item)
                    checked.remove(item)
                else:
                    checked.add(item)
        print(checked)
        
        # # simple links
        # for l in links:
        #     line1, line2 = [str(i) for i in l]
        #     x1, y1, x2, y2 = np.array(annotations_map[line1]['box'])
        #     center1 = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        #     x1, y1, x2, y2 = np.array(annotations_map[line2]['box'])
        #     center2 = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        #     cv2.line(text_mask, center1, center2, magenta, thickness=3)
        #     # draw_link(text_mask, center1, center2, colour=magenta, thinkness=6)
        
        # key-value word-to-word links for training
        horizontal_relation_mask, vertical_relation_mask = \
            draw_relation(text_mask.shape, annotations_map, colour=magenta)
        
        text_mask = np.maximum.reduce([vertical_relation_mask, 
                                       horizontal_relation_mask, 
                                       text_mask])
        
        blended = cv2.addWeighted(image, 0.6, text_mask, 0.4, 0.0)
        
        return blended
    
    image_map = {get_file_name(p): p 
                 for p in paths.list_images(image_dir)}
    label_map = {get_file_name(p): p 
                 for p in paths.list_files(label_dir, validExts=('.json'))}
    
    save_path = Path(out_dir)
    save_path.mkdir(exist_ok=True)
    for k in image_map.keys():
    # for k in ['0011906503']:
        print(image_map[k])
        try:
            image = cv2.imread(image_map[k], cv2.IMREAD_COLOR)
            annotations = read_json(label_map[k])['form']
            visualizing_image = visualize(image, annotations)
            cv2.imwrite(str(save_path.joinpath(Path(k).name).with_suffix('.png')), 
                        cv2.cvtColor(visualizing_image, cv2.COLOR_BGR2RGB))
            plt.imshow(visualizing_image)
            plt.show()
        except Exception as e:
            tb.print_exc()

if __name__ == '__main__':
    visualize_data(image_dir='dataset//training_data/images',
                   label_dir='dataset//training_data/adjusted_annotations',
                   out_dir='dataset//training_data/adjusted_relation_visualization')