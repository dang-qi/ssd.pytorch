from .config import HOME
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import h5py

class ModanetDetectionHDF5(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, hdf5_root, part='train', transform=None,
                 dataset_name='MODANET HDF5'):
        #sys.path.append(osp.join(root, COCO_API))
        from pycocotools.coco import COCO
        self.root = hdf5_root
        self.part = part
        self.h5 = h5py.File(hdf5_root, 'r')
        self.image = self.h5[part]
        self.keys = list(self.image.keys())
        #self.coco = COCO(osp.join(root, ANNOTATIONS,
        #                          INSTANCES_SET.format(image_set)))
        #self.catIds = self.coco.getCatIds(catNms=['person'])
        #self.ids = self.coco.getImgIds(catIds=self.catIds)
        #self.ids = list(self.coco.imgToAnns.keys())
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
        return len(self.keys)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        image = self.image[self.keys[index]]
        img = np.array(image['data'])
        #ori_img = Image.fromarray(img.astype(np.uint8))
        boxes = np.array(image['bbox']).astype(np.float32)
        crop_box = np.array(image['crop_box'])
        labels = np.array(image['category_id'])-1
        image_id = np.array(image['image_id'])
        resize_scale = np.array(image['scale'])
        mirrored = image['mirrored'][()]
        height, width, _ = img.shape
        inputs = {}

        # transform labels 
        # transfer targets to [x1, y1, x2, y2, label] (normalized to 1)
        scale = np.array([width, height, width, height])
        boxes = boxes / scale
        target = np.column_stack((boxes, labels))
        if self.transform is not None:
            #img = img.transpose((2,1,0)) #make it GBR mode
            img = img[:, :, (2,1,0)] # make it BGR
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        inputs['data'] = torch.from_numpy(img).permute(2, 0, 1)
        inputs['height'] = height
        inputs['width'] = width
        inputs['image_id'] = image_id
        inputs['crop_box'] = crop_box
        inputs['scale'] = resize_scale
        return inputs, target

    def get_img_id(self, index):
        return self.ids[index]

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
