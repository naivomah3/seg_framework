import os
import cv2
import numpy as np


class DataCreator:
    """
    Classes for data loading and preprocessing
    Args:
        images_path (str): path to images folder
        masks_path (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. normalization, shape manipulation, etc.)
    """

    CLASSES = ['background', 'vessels']

    def __init__(
            self,
            images_path=None,
            masks_path=None,
            classes=None,           # list of classes
            augmentation=None,      # Object: augmentation function
            preprocessing=None,     # Object: preprocessing function
            prediction=False,
    ):

        self.ids = os.listdir(images_path)
        self.images_fps = [os.path.join(images_path, image_id) for image_id in self.ids]
        self.predict = prediction
        if not self.predict: # mask not needed for prediction
            self.masks_fps = [os.path.join(masks_path, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        # load functions
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.predict:
            # extract certain classes from mask (e.g. vessels)
            mask = cv2.imread(self.masks_fps[i], 0)
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')

            # add background if mask is not binary
            if mask.shape[-1] != 1:
                background = 1 - mask.sum(axis=-1, keepdims=True)
                mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            if not self.predict:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            else:
                sample = self.augmentation(image=image)
                image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            if not self.predict:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
                return image, mask
            else:
                sample = self.preprocessing(image=image)
                image = sample['image']
                return image

    def __len__(self):
        return len(self.ids)

