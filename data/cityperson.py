__all__ = ['CityPersonTrainset', 'CityPersonTestset', 'CITYPERSON_BBOX_LABEL_NAMES']
import json
import os

import numpy as np
#from mxtorch.vision.bbox_tools import resize_bbox
#from data.dataset import caffe_normalize
from data.util import read_image, resize_bbox


def get_valid_data(annotation_path, img_path):
    """Get all valid images and annotations, which contain people.
    Args:
        annotation_path: annotation path
        img_path: image path
    Returns:
        valid annotation list and image list.
    """
    annotation_list, img_list = list(), list()
    city_list = sorted(os.listdir(annotation_path))
    for city in city_list:
        city_dir = os.path.join(annotation_path, city)
        data_list = sorted(os.listdir(city_dir))
        for a in data_list:
            annot_path = os.path.join(city_dir, a)
            with open(annot_path, 'r') as f:
                annot_ = json.load(f)
            valid_index = 0
            for i in annot_['objects']:
                if i['label'] != 'ignore':
                    valid_index += 1
            if valid_index > 0:
                annotation_list += [os.path.join(city_dir, a)]
                
                img_name_ = a.split('.')[0].split('_')[:-1]
                img_name = ''
                for n in img_name_:
                    img_name += (n + '_')
                img_name += 'leftImg8bit.png'
                img_list += [os.path.join(img_path, city, img_name)]
    return annotation_list, img_list

def get_valid_data2(annotation_path, img_path):
    """Get all valid images and annotations, which contain people.
    Args:
        annotation_path: annotation path
        img_path: image path
    Returns:
        valid annotation list and image list.
    """
    annotation_list, img_list, image_id_list = list(), list(), list()
    count = 0
    city_list = sorted(os.listdir(annotation_path))
    for city in city_list:
        city_dir = os.path.join(annotation_path, city)
        data_list = sorted(os.listdir(city_dir))
        for a in data_list:
            annot_path = os.path.join(city_dir, a)
            with open(annot_path, 'r') as f:
                annot_ = json.load(f)
            count += 1
            valid_index = 0
            for i in annot_['objects']:
                if i['label'] != 'ignore':
                    valid_index += 1
            if valid_index > 0:
                annotation_list += [os.path.join(city_dir, a)]
                img_name_ = a.split('.')[0].split('_')[:-1]
                img_name = ''
                for n in img_name_:
                    img_name += (n + '_')
                img_name += 'leftImg8bit.png'
                img_list += [os.path.join(img_path, city, img_name)]
                image_id_list.append(count)
    return annotation_list, img_list, image_id_list

class CityPersonTrainset:
    def __init__(self, img_path, annotation_path):
        self.annotation_list, self.img_list = get_valid_data(annotation_path, img_path)
        self.label_names = CITYPERSON_BBOX_LABEL_NAMES
        
    def get_example(self, item):
        # Get origin image.
        img_name = self.img_list[item]
        ori_img = read_image(img_name)
        difficult = list()
        # Get bounding boxes annotation.
        annotation = self.annotation_list[item]
        with open(annotation, 'r') as f:
            annot = json.load(f)
        bbox_list = list()
        for i in annot['objects']:
            if i['label'] != 'ignore':
                x, y, w, h = i['bbox']
                bbox_list += [[y-1, x-1, y-1 + h, x-1 + w]]
        bbox = np.stack(bbox_list).astype(np.float32)

        # Get label.
        label = np.zeros(bbox.shape[0], dtype=np.int32)
        
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        return ori_img, bbox, label, difficult

    def __len__(self):
        return len(self.img_list)

    __getitem__ = get_example


class CityPersonTestset(object):
    def __init__(self, img_path, annotation_path):
        self.annotation_list, self.img_list, self.image_id_list = get_valid_data2(annotation_path, img_path)
        self.label_names = CITYPERSON_BBOX_LABEL_NAMES

    def get_example(self, item):
        img_name = self.img_list[item]
        image_id = self.image_id_list[item]
        ori_img = read_image(img_name)
        annotation = self.annotation_list[item]
        with open(annotation, 'r') as f:
            annot = json.load(f)
        bbox_list = list()
        for i in annot['objects']:
            if i['label'] != 'ignore':
                x, y, w, h = i['bbox']
                bbox_list += [[y-1, x-1, y-1 + h, x-1 + w]]

        bbox = np.stack(bbox_list).astype(np.float32)

        # Get label.
        label = np.zeros(bbox.shape[0], dtype=np.int32)

        # Get difficult.
        difficult = np.zeros(label.shape, dtype=np.uint8)

        return ori_img, bbox, label, difficult, image_id
        
    def __len__(self):
        return len(self.img_list)

    __getitem__ = get_example

CITYPERSON_BBOX_LABEL_NAMES = (
    'pedestrian',
    'riders',
    'sitting persons',
    'other persons with unusual postures',
    'group of people'
)
