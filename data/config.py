# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

coco_person_416 = {
    'num_classes': 2, # include the background
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    #'feature_maps': [38, 19, 10, 5, 3, 1],
    'feature_maps': [52, 26, 13, 7, 5, 3],
    'min_dim': 416,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO_PERSON_416',
}

coco_person_800 = {
    'num_classes': 2, # include the background
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    #'feature_maps': [38, 19, 10, 5, 3, 1],
    'feature_maps': [100, 50, 25, 13, 11, 9],
    'min_dim': 800,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO_PERSON_800',
}
modanet = {
    'num_classes': 14,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'MODANET',
}

modanet_256 = {
    'num_classes': 14,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [32, 16, 8, 4, 2], # might be wrong
    'min_dim': 256,
    'steps': [8, 16, 32, 64, 128 ],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'MODANET',
}

modanet_128 = {
    'num_classes': 14,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [16, 8, 4, 2 ], # might be wrong
    'min_dim': 128,
    'steps': [8, 16, 32, 64],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'MODANET',
}

modanet_512 = {
    'num_classes': 14,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [64, 32, 16, 8, 6, 4],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 86, 128],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'MODANET',
}


modanet_whole_800 = {
    'num_classes': 14, # include the background
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    #'feature_maps': [38, 19, 10, 5, 3, 1],
    'feature_maps': [100, 50, 25, 13, 11, 9],
    'min_dim': 800,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'MODANET_WHOLE_800',
}