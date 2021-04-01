import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.transforms import Normalize

import albumentations as A
from albumentations.pytorch import ToTensorV2
from training.auto_augment import AutoAugment, Cutout

from timm.data.transforms_factory import transforms_imagenet_train
from timm.data.random_erasing import RandomErasing

# imagenet
normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])

# vit norm
# normalize = Normalize(mean=[0.5, 0.5, 0.5],
#                       std=[0.5, 0.5, 0.5])

CROP_HEIGHT = 512  # 380
CROP_WIDTH = 512

_, rand_augment, _ = transforms_imagenet_train((CROP_HEIGHT, CROP_WIDTH),
                                               auto_augment='original-mstd0.5',
                                               separate=True)


# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

        # https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312/9
        img = transforms.ToTensor()(img).unsqueeze_(0)
        _, _, h, w = img.size()
        # print(img.size())

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        # https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312/9
        return transforms.ToPILImage()(img.squeeze_(0))


def training_augmentation3():
    # train_transform = [
    #     # transforms.Resize((CROP_HEIGHT, CROP_WIDTH)),
    #     transforms.CenterCrop((CROP_HEIGHT, CROP_WIDTH)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.1, contrast=0.1, ),
    #     # AutoAugment(),
    #     transforms.ToTensor(),
    #     # imagenet norm
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    #     Cutout(n_holes=8, length=10),
    # ]
    # return transforms.Compose(train_transform)

    train_transform = [
        A.Resize(CROP_HEIGHT, CROP_WIDTH),
        # A.OneOf([
        #     A.Resize(CROP_HEIGHT, CROP_WIDTH),
        #     A.CenterCrop(CROP_HEIGHT, CROP_WIDTH),
        #     A.RandomResizedCrop(CROP_HEIGHT, CROP_WIDTH)
        # ], p=1.),
        A.Transpose(p=0.5),
        A.HorizontalFlip(),
        # A.VerticalFlip(),
        # A.OneOf([
        #     A.IAAAdditiveGaussianNoise(),
        #     A.GaussNoise(),
        # ], p=0.3),
        # A.OneOf([
        #     A.MotionBlur(p=0.25),
        #     A.GaussianBlur(p=0.5),
        #     A.Blur(blur_limit=3, p=0.25),
        # ], p=0.3),
        # A.Blur(blur_limit=3, p=0.3),
        # A.RandomGamma(p=0.5),
        # A.RandomBrightnessContrast(brightness_limit=0.2, p=0.3),
        # # A.HueSaturationValue(p=0.5),
        # A.ShiftScaleRotate(p=0.3),
        # A.CoarseDropout(max_holes=4, p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)


# def training_augmentation2():
#     train_transform = [
#         A.CenterCrop(CROP_HEIGHT, CROP_WIDTH),
#         A.Flip(),
#         # A.RandomCrop(CROP_HEIGHT, CROP_WIDTH),
#         VisionTransform(rand_augment, is_tensor=False, p=0.5),
#         A.Normalize(),
#         ToTensorV2(),
#     ]
#     return A.Compose(train_transform)


def validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    # validation_transform = [
    #     transforms.Resize((CROP_HEIGHT, CROP_WIDTH)),
    #     # transforms.CenterCrop((CROP_HEIGHT, CROP_WIDTH)),
    #     # transforms.ToTensor(),
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #     #                      std=[0.229, 0.224, 0.225]),
    #     # TTA
    #     FiveCrop(320),
    #     # returns a 4D tensor
    #     Lambda(lambda crops: torch.stack([normalize(ToTensor()(crop)) for crop in crops])),
    # ]
    # return transforms.Compose(validation_transform)

    validation_transform = [
        # A.Resize(600, 800),
        # A.CenterCrop(CROP_HEIGHT, CROP_WIDTH),
        A.Resize(CROP_HEIGHT, CROP_WIDTH),
        # A.ShiftScaleRotate(shift_limit=0.0, scale_limit=(0.4, 0.5), p=0.7),
        A.Normalize(),
        # A.Normalize(mean=(0.43, 0.497, 0.313), std=(0.219, 0.224, 0.201)),
        ToTensorV2(),
    ]
    return A.Compose(validation_transform)


def test_augmentation():
    """Add paddings to make image shape divisible by 32"""
    # test_transform = [
    #     transforms.Resize((CROP_HEIGHT, CROP_WIDTH)),
    #     # transforms.CenterCrop((448, 448)),
    #     transforms.ToTensor(),
    #     # test norm
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    #     # # TTA
    #     # TenCrop(384),
    #     # # returns a 4D tensor
    #     # Lambda(lambda crops: torch.stack([normalize(ToTensor()(crop)) for crop in crops])),
    # ]
    # return transforms.Compose(test_transform)

    test_transform = [
        A.Resize(CROP_HEIGHT, CROP_WIDTH),
        A.Normalize(),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)


def verify_augmentation():
    train_transform = [
        A.Resize(CROP_HEIGHT, CROP_WIDTH),
        A.HorizontalFlip(),
        A.VerticalFlip(p=0.1),
        A.OneOf([
            A.MultiplicativeNoise(multiplier=(0.7, 1.2), elementwise=True),
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(var_limit=100),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.3),
            A.GaussianBlur(p=0.5),
            A.Blur(blur_limit=3, p=0.3),
        ], p=0.3),
        A.HueSaturationValue(p=0.2),
        # A.RandomCrop(CROP_HEIGHT, CROP_WIDTH),
        # VisionTransform(rand_augment, is_tensor=False, p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)
