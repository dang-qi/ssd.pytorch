"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform, COCOPersonDetection, COCO_ROOT
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import json
import cv2
import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from data import detection_collate
import warnings
warnings.filterwarnings("ignore")

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_arg():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval/', type=str,
                        help='File path to save results')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to train model')
    parser.add_argument('--voc_root', default=VOC_ROOT,
                        help='Location of VOC root directory')
    parser.add_argument('--coco_root', default=COCO_ROOT,
                        help='Location of VOC root directory')
    parser.add_argument('--cleanup', default=True, type=str2bool,
                        help='Cleanup and remove results files following eval')
    parser.add_argument('--just_person', default=False, type=str2bool,
                        help='Just evaluate person result for coco')
    parser.add_argument('--dataset', default='coco_person', type=str,
                        help='Datset name, can be coco_person and modanet')

    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't using \
                CUDA.  Run with --cuda for optimal eval speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    
    return args

#annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
#imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
#imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
#                          'Main', '{:s}.txt')
#YEAR = '2007'
#devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def test_net_coco(anno_file, net, cuda, dataloader, transform, just_person=False,
             im_size=300, thresh=0.05):
    '''
    evaluation format:
    [{
        "image_id": int, 
        "category_id": int, 
        "bbox": [x,y,width,height], 
        "score": float
    }]
    '''
    #num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    #all_boxes = [[[] for _ in range(num_images)]
    #             for _ in range(len(labelmap)+1)]

    # timers
    #_t = {'im_detect': Timer(), 'misc': Timer()}

    print('start evaluate')
    net.eval()
    detection_time = 0
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.json')

    results = []
    data_num = 0
    #for i in range(num_images):
    for i, (inputs, gt) in enumerate(tqdm.tqdm(dataloader, desc='Testing')):
        #im, gt, h, w = dataset.pull_item(i)

        #x = Variable(im.unsqueeze(0))

        x = inputs['data']
        data_num +=len(x)
        if cuda:
            x = x.cuda()
        #_t['im_detect'].tic()
        start = time.time()
        detections = net(x).data
        detection_time += time.time()-start
        #detect_time = _t['im_detect'].toc(average=False)

        for detection, h, w, im_id in zip(detections, inputs['height'], inputs['width'], inputs['image_id']):
            for j, dets in enumerate(detection): 
                # skip j = 0, because it's the background class
                if j==0:
                    continue
            #dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()

                boxes = boxes.cpu().numpy().astype(int)
                for box, score in zip(boxes, scores):
                    results.append({'image_id': im_id,
                                    'category_id': j,
                                    'bbox': box.tolist(),
                                    'score': float(score)})

    print('total detection time is {}'.format(detection_time))
    print('detection time per image is {}'.format(detection_time/data_num))
    print('Evaluating detections')
    run_eval_coco(results, anno_file, det_file, just_person=just_person)

def run_eval_coco(results, anno_file, det_file, just_person=False):
    with open(det_file, 'w') as f:
        json.dump(results, f)
    coco = COCO(anno_file)
    coco_dets = coco.loadRes(det_file)
    coco_eval = COCOeval(coco, coco_dets, "bbox")
    if just_person:
        print('Just test person category')
        coco_eval.params.catIds = [1] # just run evaluation on person category
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    args = parse_arg()
    if args.dataset == 'coco_person':
        gt_json=os.path.expanduser('~/data/datasets/COCO/annotations/instances_val2014.json')
        num_classes = 2
    elif args.dataset == 'modanet':
        num_classes = 14 # plus background
        pass
    else:
        raise TypeError('Invalid dataset name')
    # load net
    #num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = COCOPersonDetection(args.coco_root, image_set='val2014', transform=BaseTransform(300, dataset_mean))
    data_num = len(dataset)
    #dataset = VOCDetection(args.voc_root, [('2007', set_type)],
    #                       BaseTransform(300, dataset_mean),
    #                       VOCAnnotationTransform())
    data_loader = data.DataLoader(dataset, batch_size=1,
                                  num_workers=8,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net_coco( gt_json, net, args.cuda, data_loader,
                  BaseTransform(net.size, dataset_mean), im_size=300,
                  just_person=args.just_person, thresh=args.confidence_threshold)
