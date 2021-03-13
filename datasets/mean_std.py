import os
import cv2
import numpy as np
from tqdm import tqdm

import transformer
from datasets.cassava import CassavaDataset


# # count = 0
# # mean = 0
# # delta = 0
# # delta2 = 0
# # M2 = 0
#
# r_channel_mean = []
# g_channel_mean = []
# b_channel_mean = []
#
# r_channel_std = []
# g_channel_std = []
# b_channel_std = []
#
# src = '/home/ace19/dl_data/shopee-product-matching/train_images'
# sources = os.listdir(src)
#
# for i, file in enumerate(tqdm(sources)):
#     img_name = os.path.join(src, file)
#     # BGR
#     image = cv2.imread(img_name, cv2.IMREAD_COLOR)
#     # blue
#     b_val = np.reshape(image[:, :, 0], -1)
#     # green
#     g_val = np.reshape(image[:, :, 1], -1)
#     # red
#     r_val = np.reshape(image[:, :, 2], -1)
#
#     img_r_mean = np.mean(r_val) / 255
#     img_g_mean = np.mean(g_val) / 255
#     img_b_mean = np.mean(b_val) / 255
#     r_channel_mean.append(img_r_mean)
#     g_channel_mean.append(img_g_mean)
#     b_channel_mean.append(img_b_mean)
#
#     img_r_std = np.std(r_val) / 255
#     img_g_std = np.std(g_val) / 255
#     img_b_std = np.std(b_val) / 255
#     r_channel_std.append(img_r_std)
#     g_channel_std.append(img_g_std)
#     b_channel_std.append(img_b_std)
#
# print('r mean: ', round(sum(r_channel_mean) / len(r_channel_mean), 3))
# print('g mean: ', round(sum(g_channel_mean) / len(g_channel_mean), 3))
# print('b mean: ', round(sum(b_channel_mean) / len(b_channel_mean), 3))
#
# print('r std: ', round(sum(r_channel_std) / len(r_channel_std), 3))
# print('g std: ', round(sum(g_channel_std) / len(g_channel_std), 3))
# print('b std: ', round(sum(b_channel_std) / len(b_channel_std), 3))

# https://github.com/facebookresearch/mixup-cifar10/blob/master/utils.py
import torch


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')

    tbar = tqdm(dataloader, desc='\r')
    for inputs, _, _, _ in tbar:
        # inputs = inputs.cuda()
        inputs = inputs.float()

        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


trainset = CassavaDataset(data_dir='/home/ace19/dl_data/shopee-product-matching',
                          fold=['train_all_27053.npy'],
                          csv=['train.csv'],
                          mode='train',
                          transform=None,
                          # transform = transformer.training_augmentation2(),
                          # transform=transformer.training_augmentation(),
                          )

mean, std = get_mean_and_std(trainset)
print(mean)
print(std)
