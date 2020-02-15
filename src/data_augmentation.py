import random
import numpy as np
import cv2
# albumentations
import albumentations as alb

#------------------------------------------------
# Online augmentations (using Albumentations)
#------------------------------------------------
def get_data_train_aug(in_h, in_w):
    """
    Add paddings to make image shape divisible by 32
    Further Albumentations effect can be added here
    """
    train_transform = [
        #alb.RandomCrop(height=320, width=320, always_apply=True),
        alb.PadIfNeeded(in_h, in_w)
    ]
    return alb.Compose(train_transform)


def get_data_valid_aug(in_h, in_w):
    """
    Add paddings to make image shape divisible by 32
    """
    validation_transform = [
        #alb.RandomCrop(height=320, width=320, always_apply=True),
        alb.PadIfNeeded(in_h, in_w)
    ]
    return alb.Compose(validation_transform)


def get_data_test_aug(in_h, in_w):
    """
    Add paddings to make image shape divisible by 32
    """
    test_transform = [
        #alb.RandomCrop(height=320, width=320, always_apply=True),
        alb.PadIfNeeded(in_h, in_w)
    ]
    return alb.Compose(test_transform)


def get_data_pred_aug(in_h, in_w):
    """
    Add paddings if not matching the desired dimension: waste of time :D
    """
    pred_transform = [
        alb.PadIfNeeded(in_h, in_w)
    ]
    return alb.Compose(pred_transform)

def get_preprocessing(preprocessing_fn):
    """
    Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    preprocess_transform = [
        alb.Lambda(image=preprocessing_fn),
    ]
    return alb.Compose(preprocess_transform)

#------------------------------------------------
# Offline: light augmentations (using OpenCV)
#------------------------------------------------
def apply_rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rotation, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def apply_random_noise(image):
    height, width, channel = image.shape
    mat = np.random.randn(height, width, channel) * random.randint(5, 30)
    return np.clip(image+mat, 0, 255).astype(np.uint8)

def apply_change_gamma(image, alpha=1.0, beta=0.0):
    return np.clip(alpha*image+beta, 0, 255).astype(np.uint8)

def apply_random_color(image, alpha=20):
    mat = [random.randint(-alpha, alpha), random.randint(-alpha, alpha),random.randint(-alpha, alpha)]
    return np.clip(image+mat, 0, 255).astype(np.uint8)

def apply_random_transformation(image):
    if np.random.randint(2):
        image = apply_change_gamma(image, random.uniform(0.8, 1.2), np.random.randint(100)-50)
    if np.random.randint(2):
        image = apply_random_noise(image)
    if np.random.randint(2):
        image = apply_random_color(image)
    return image