from .voc0712 import VOCDetection, VOCAnnotationTransform, VOC_CLASSES, VOC_ROOT

from .coco import COCODetection, COCOAnnotationTransform, COCO_CLASSES, COCO_ROOT, get_label_map
from .coco_person import COCOPersonDetection, COCO_PERSON_CLASSES
from .modanet_hdf5 import ModanetDetectionHDF5
from .modanet import ModanetDetection
from .config import *
import torch
import cv2
import numpy as np

configs = {}
configs['coco_person_416'] = coco_person_416
configs['coco_person_800'] = coco_person_800
configs['modanet_128'] = modanet_128
configs['modanet_256'] = modanet_256
configs['modanet_512'] = modanet_512
configs['modanet_whole_800'] = modanet_whole_800


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    inputs = {}
    ws = []
    hs = []
    im_ids = []
    scales = []
    crop_box = []
    for sample in batch:
        imgs.append(sample[0]['data'])
        targets.append(torch.FloatTensor(sample[1]))
        ws.append(sample[0]['width'])
        hs.append(sample[0]['height'])
        im_ids.append(sample[0]['image_id'])
        if 'scale' in sample[0]:
            scales.append(sample[0]['scale'])
        if 'crop_box' in sample[0]:
            crop_box.append(sample[0]['crop_box'])
    inputs['data'] = torch.stack(imgs, 0)
    inputs['width'] = ws
    inputs['height'] = hs
    inputs['image_id'] = im_ids
    inputs['scale'] = scales
    inputs['crop_box'] = crop_box
    return inputs, targets


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels
