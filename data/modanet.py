from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import pickle


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map(osp.join(COCO_ROOT, 'coco_labels.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class ModanetDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, anno_root, im_root, part='train', transform=None,
                 dataset_name='MODANET'):
        with open(anno_root, 'rb') as f:
            self.images = pickle.load(f)[part]
        self.im_root = im_root
        self.part = part
        self.transform = transform
        self.name = dataset_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        inputs, gt = self.pull_item(index)
        return inputs, gt

    def __len__(self):
        return len(self.images)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        image = self.images[index]
        im_path = os.path.join(self.im_root, image['file_name'])
        image_id = image['id']

        img = cv2.imread(im_path)
        height, width, _ = img.shape

        if self.transform is not None:
            if len(image['objects'])>0:
                boxes = np.zeros((len(image['objects']), 4), dtype=float)
                labels = np.zeros((len(image['objects'])), dtype=float)
                for i, a_object in enumerate(image['objects']):
                    labels[i] = a_object['category_id'] - 1
                    boxes[i, 0] = a_object['bbox'][0]
                    boxes[i, 1] = a_object['bbox'][1]
                    boxes[i, 2] = a_object['bbox'][0] + a_object['bbox'][2]
                    boxes[i, 3] = a_object['bbox'][1] + a_object['bbox'][3]
                # normalize the (x,y,x,y) to (0,1)
                scale = np.array([width, height, width, height])
                boxes = boxes / scale

                img, boxes, labels = self.transform(img, boxes, labels)
                # to rgb
                img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        inputs = {}
        inputs['data'] = torch.from_numpy(img).permute(2, 0, 1)
        inputs['height'] = height
        inputs['width'] = width
        inputs['image_id'] = image_id
        #inputs['scale'] = resize_scale


        return inputs, target

    #def get_img_id(self, index):
    #    return self.ids[index]

    #def pull_image(self, index):
    #    '''Returns the original image object at index in PIL form

    #    Note: not using self.__getitem__(), as any transformations passed in
    #    could mess up this functionality.

    #    Argument:
    #        index (int): index of img to show
    #    Return:
    #        cv2 img
    #    '''
    #    img_id = self.ids[index]
    #    path = self.coco.loadImgs(img_id)[0]['file_name']
    #    return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    #def pull_anno(self, index):
    #    '''Returns the original annotation of image at index

    #    Note: not using self.__getitem__(), as any transformations passed in
    #    could mess up this functionality.

    #    Argument:
    #        index (int): index of img to get annotation of
    #    Return:
    #        list:  [img_id, [(label, bbox coords),...]]
    #            eg: ('001718', [('dog', (96, 13, 438, 332))])
    #    '''
    #    img_id = self.ids[index]
    #    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    #    return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.img_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
