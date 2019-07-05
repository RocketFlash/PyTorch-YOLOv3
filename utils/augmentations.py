import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from albumentations import torch as AT
from albumentations import (
    OneOf,
    Blur,
    RandomGamma,
    HueSaturationValue,
    RGBShift,
    RandomBrightness,
    RandomContrast,
    MedianBlur,
    CLAHE,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Compose,
    GaussNoise
)

def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose([aug, AT.ToTensor()], bbox_params={'format': 'albumentations',
                                     'min_area': min_area,
                                     'min_visibility': min_visibility,
                                     'label_fields': ['labels']})


def make_augmented(images, targets):
    aug_params = [OneOf([
        Blur(blur_limit=5, p=1.),
        RandomGamma(gamma_limit=(50, 150), p=1.),
        HueSaturationValue(hue_shift_limit=20,
                           sat_shift_limit=30, val_shift_limit=20, p=1.),
        RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=15, p=1.),
        RandomBrightness(limit=.25, p=1.),
        RandomContrast(limit=.25, p=1.),
        MedianBlur(blur_limit=5, p=1.),
        # CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.)
    ], p=1.),
        HorizontalFlip(p=0.5),
        GaussNoise(var_limit=(10.0, 120.0), p=0.4)
    ]

    labels = targets[:, 1]
    bboxes = targets[:, 2:]
    bboxes_new = torch.zeros((len(bboxes), 4))
    bboxes_new[:,0] = (bboxes[:, 0] - bboxes[:, 2] / 2)
    bboxes_new[:,1] = (bboxes[:, 1] - bboxes[:, 3] / 2)
    bboxes_new[:,2] = (bboxes[:, 0] + bboxes[:, 2] / 2)
    bboxes_new[:,3] = (bboxes[:, 1] + bboxes[:, 3] / 2)

    annotations = {'image': images,
                   'bboxes': bboxes,
                   'labels': labels}
    aug = get_aug(aug_params)
    augmented = aug(**annotations)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_labels = augmented['labels']

    bboxes[:, 0] = ((bboxes_new[:,0] + bboxes_new[:,2]) / 2)
    bboxes[:, 1] = ((bboxes_new[:,1] + bboxes_new[:,3]) / 2) 

    images = augmented_image
    targets[:, 1] = augmented_labels
    targets[:, 2:] = bboxes 
    return images, targets
