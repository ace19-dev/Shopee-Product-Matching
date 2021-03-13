import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

import torch


def five_crop(images, size):
    height, width = images.size()[2], images.size()[3]
    assert size < height, 'crop-size must smaller than height' % (height)
    assert size < width, 'crop-size must smaller than width' % (width)

    # img = images[0].cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))
    # # scipy.misc.toimage(img).show() Or
    # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # # cv2.imwrite('/home/ace19/Pictures/original', img)
    # plt.imsave('/home/ace19/Pictures/original.jpg', img)

    c = (height // 2, width // 2)
    # print(c[0], c[1])

    img1 = images[:, :, 0:size, 0:size]
    # print(img1.size())
    # img = img1[0].cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))
    # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # plt.imsave('/home/ace19/Pictures/img1.jpg', img)

    img2 = images[:, :, 0:size, width - size:width]
    # print(img2.size())
    # img = img2[0].cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))
    # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # plt.imsave('/home/ace19/Pictures/img2.jpg', img)

    img3 = images[:, :, height - size:height, width - size:width]
    # print(img3.size())
    # img = img3[0].cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))
    # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # plt.imsave('/home/ace19/Pictures/img3.jpg', img)

    img4 = images[:, :, height - size:height, 0:size]
    # print(img4.size())
    # img = img4[0].cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))
    # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # plt.imsave('/home/ace19/Pictures/img4.jpg', img)

    img5 = images[:, :, c[0] - (size // 2):c[0] + (size // 2), c[1] - (size // 2):c[1] + (size // 2)]
    # print(img5.size())
    # img = img5[0].cpu().numpy()
    # img = np.transpose(img, (1, 2, 0))
    # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    # plt.imsave('/home/ace19/Pictures/img5.jpg', img)

    return torch.stack([img1, img2, img3, img4, img5])


# TODO: crop 방식 다양화
def crop_images(images, size, num_samples):
    height, width = images.size()[2], images.size()[3]

    view_images = []
    for image in images:
        assert size < height, 'crop-size must smaller than height' % (height)
        assert size < width, 'crop-size must smaller than width' % (width)

        # img = images[0].cpu().numpy()
        # img = np.transpose(img, (1, 2, 0))
        # # scipy.misc.toimage(img).show() Or
        # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # # cv2.imwrite('/home/ace19/Pictures/original', img)
        # plt.imsave('/home/ace19/Pictures/original.jpg', img)

        # c = (height // 2, width // 2)
        # print(c[0], c[1])

        img1 = image[:, 0:size, 0:size]
        # print(img1.size())
        # img = img1[0].cpu().numpy()
        # img = np.transpose(img, (1, 2, 0))
        # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # plt.imsave('/home/ace19/Pictures/img1.jpg', img)

        img2 = image[:, 0:size, width - size:width]
        # print(img2.size())
        # img = img2[0].cpu().numpy()
        # img = np.transpose(img, (1, 2, 0))
        # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # plt.imsave('/home/ace19/Pictures/img2.jpg', img)

        img3 = image[:, height - size:height, width - size:width]
        # print(img3.size())
        # img = img3[0].cpu().numpy()
        # img = np.transpose(img, (1, 2, 0))
        # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # plt.imsave('/home/ace19/Pictures/img3.jpg', img)

        img4 = image[:, height - size:height, 0:size]
        # print(img4.size())
        # img = img4[0].cpu().numpy()
        # img = np.transpose(img, (1, 2, 0))
        # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # plt.imsave('/home/ace19/Pictures/img4.jpg', img)
        cropped_images = [img1, img2, img3, img4]
        samples = random.sample(cropped_images, num_samples)

        views = torch.stack(samples)
        view_images.append(views)

    return torch.stack(view_images)
