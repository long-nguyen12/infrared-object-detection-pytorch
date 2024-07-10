import os
import os.path as osp
import random
import shutil
import sys

import cv2
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np


class Dataset(Data.Dataset):
    def __init__(self, img_paths, mask_paths, transform=None, mask_transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)

        if self.transform is not None:
            img_aug = self.transform(image=image)
            image = img_aug["image"]

        if self.mask_transform is not None:
            mask_aug = self.mask_transform(image=mask)
            mask = mask_aug["image"]

        image = image.astype("float32") / 255
        image = image.transpose((2, 0, 1))

        mask = mask[:, :, np.newaxis]
        mask = mask.astype("float32") / 255
        mask = mask.transpose((2, 0, 1))

        return np.asarray(image), np.asarray(mask)

    def __len__(self):
        return len(self.img_paths)
