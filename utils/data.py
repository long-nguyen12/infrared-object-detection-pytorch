import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import os
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import shutil
from glob import glob


class IRSTD_Dataset(Data.Dataset):
    def __init__(self, args, mode="train", dataset=None):

        dataset_dir = args.train_path
        self.dataset = dataset
        if mode == "train":
            txtfile = "trainval.txt"
        elif mode == "val":
            txtfile = "test.txt"

        self.list_dir = osp.join(dataset_dir, txtfile)
        self.imgs_dir = osp.join(dataset_dir, "images")
        self.label_dir = osp.join(dataset_dir, "masks")

        self.names = []
        with open(self.list_dir, "r") as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        name = self.names[i]
        if self.dataset == "SIRST":
            mask_name = "_pixels0.png"
        else:
            mask_name = ".png"

        img_path = osp.join(self.imgs_dir, name + ".png")
        label_path = osp.join(self.label_dir, name + mask_name)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(label_path)

        if self.mode == "train":
            img, mask = self._sync_transform(img, mask)
        elif self.mode == "val":
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask


class SIRST_Dataset(Data.Dataset):
    def __init__(self, args, mode="train"):

        dataset_dir = args.train_path

        if mode == "train":
            dataset_dir = osp.join(dataset_dir, "training")
        elif mode == "val":
            dataset_dir = osp.join(dataset_dir, "test")

        self.imgs_dir = osp.join(dataset_dir, "images")
        self.label_dir = osp.join(dataset_dir, "masks")

        self.names = []
        self.mask_names = []

        self.names = glob("{}/*".format(self.imgs_dir))
        self.mask_names = glob("{}/*".format(self.label_dir))

        self.names.sort()
        self.mask_names.sort()

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        name = self.names[i]
        label_name = self.mask_names[i]

        img = Image.open(name).convert("RGB")
        mask = Image.open(label_name)

        if self.mode == "train":
            img, mask = self._sync_transform(img, mask)
        elif self.mode == "val":
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return img, mask

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask
