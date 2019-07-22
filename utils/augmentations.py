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

import imgaug.augmenters as iaa

import sys
import cv2
from matplotlib import pyplot as plt
import time
##########################FOR DEBUG###################################################
BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=5):

    x1, y1, x2, y2 = bbox
    h_i, w_i, c_i = img.shape
    x_min = x1*w_i
    y_min = y1*h_i
    w = (x2-x1) * w_i
    h = (y2-y1) * h_i
    x_min, x_max, y_min, y_max = int(x_min), int(
        x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name, img_name):

    category_id_to_name = {0: 'car',
                           1: 'bus',
                           2: 'person',
                           3: 'bike',
                           4: 'truck',
                           5: 'motor',
                           6: 'train',
                           7: 'rider',
                           8: 'traffic sign',
                           9: 'traffic light'}

    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(
            img, bbox, annotations['labels'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imsave(img_name, img)
##########################FOR DEBUG###################################################


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug,
                   bbox_params={'format': 'albumentations',
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
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.)
    ], p=1.),
        HorizontalFlip(p=1),
        GaussNoise(var_limit=(10.0, 120.0), p=0.4)
    ]

    labels = targets[:, 1]
    bboxes = targets[:, 2:]

    bboxes_new = torch.zeros((len(bboxes), 4))
    bboxes_new[:, 0] = (bboxes[:, 0] - bboxes[:, 2] / 2)
    bboxes_new[:, 1] = (bboxes[:, 1] - bboxes[:, 3] / 2)
    bboxes_new[:, 2] = (bboxes[:, 0] + bboxes[:, 2] / 2)
    bboxes_new[:, 3] = (bboxes[:, 1] + bboxes[:, 3] / 2)

    # bboxes_new = bboxes_new.numpy()

    annotations = {'image': images,
                   'bboxes': bboxes_new.numpy(),
                   'labels': labels.numpy()}

    aug = get_aug(aug_params)
    augmented = aug(**annotations)
    augmented_image = augmented['image']
    augmented_bboxes = np.array(augmented['bboxes'])
    augmented_labels = np.array(augmented['labels'])

    bboxes[:, 0] = torch.from_numpy(
        ((augmented_bboxes[:, 0] + augmented_bboxes[:, 2]) / 2))
    bboxes[:, 1] = torch.from_numpy(
        ((augmented_bboxes[:, 1] + augmented_bboxes[:, 3]) / 2))
    bboxes[:, 2] = torch.from_numpy(
        (augmented_bboxes[:, 2] - augmented_bboxes[:, 0]))
    bboxes[:, 3] = torch.from_numpy(
        (augmented_bboxes[:, 3] - augmented_bboxes[:, 1]))
    images = torch.from_numpy(augmented_image).float()
    targets[:, 1] = torch.from_numpy(augmented_labels).float()
    targets[:, 2:] = bboxes

    # visualize(annotations, 'test_before.jpg')
    # time.sleep(1)
    # visualize(augmented,  'test_after.jpg')
    # time.sleep(10)

    return images, targets


def random_float(low, high):
    return np.random.random()*(high-low) + low


def make_augmented_night(image):
    mul = random_float(0.1, 0.5)
    add = np.random.randint(-100, -50)
    gamma = random_float(2, 3)

    aug = iaa.OneOf([
        iaa.Multiply(mul=mul),
        iaa.Add(value=add),
        iaa.contrast.GammaContrast(gamma=gamma)
    ])

    image_augmented = aug.augment_image(image)

    return image_augmented
