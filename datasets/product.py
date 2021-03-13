import os
import random
import pandas as pd
from PIL import Image

import cv2
import numpy as np
import torch.utils.data as data

# NUM_CLASS = 5


class ProductDataset(data.Dataset):
    def __init__(self, data_dir, fold, csv, mode, transform=None):
        self.data_dir = data_dir
        self.fold = fold
        self.csv = csv
        self.mode = mode
        self.transform = transform

        samples = list(np.concatenate([np.load(data_dir + '/fold/%s' % f, allow_pickle=True) for f in self.fold]))
        self.df = pd.concat([pd.read_csv(data_dir + '/%s' % f) for f in self.csv])
        self.df = df_loc_by_list(self.df, 'posting_id', samples)
        self.images = self.df['image'].values.tolist()
        self.labels = self.df['label'].values.tolist()  # create 'label' by grouping 'label_group'
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        image_id = self.images[index]
        # image_path = os.path.join(self.data_dir, 'train_images', image_id)
        image_path = os.path.join(self.data_dir, 'train_images', image_id)
        # print('##### ', image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = Image.fromarray(image)

        if self.transform is not None:
            sample = self.transform(image=image)
            image = sample['image']
            # image = self.transform(image)

        return image, self.labels[index], image_id, index

    # def __str__(self):
    #     label1 = (self.df['label'] == 0).sum()
    #     label2 = (self.df['label'] == 1).sum()
    #     label3 = (self.df['label'] == 2).sum()
    #     label4 = (self.df['label'] == 3).sum()
    #     label5 = (self.df['label'] == 4).sum()
    #
    #     length = len(self)
    #
    #     string = ''
    #     string += '\tmode  = %s\n' % self.mode
    #     string += '\tfold = %s\n' % self.fold
    #     string += '\tcsv   = %s\n' % str(self.csv)
    #
    #     string += '\t\tlen  = %5d\n' % length
    #     string += '\t\tlabel1 = %5d, label1/length = %0.3f\n' % (label1, label1 / length)
    #     string += '\t\tlabel2 = %5d, label2/length = %0.3f\n' % (label2, label2 / length)
    #     string += '\t\tlabel3 = %5d, label3/length = %0.3f\n' % (label3, label3 / length)
    #     string += '\t\tlabel4 = %5d, label4/length = %0.3f\n' % (label4, label4 / length)
    #     string += '\t\tlabel5 = %5d, label5/length = %0.3f\n' % (label5, label5 / length)
    #
    #     return string

    def __len__(self):
        return len(self.images)

    def fold_name(self):
        return self.fold[0].split('_')[1]


class ProductTestDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # self.images = self.read_test_dataset()

        self.df = pd.concat([pd.read_csv(data_dir + '/%s' % f) for f in ['sample_submission.csv']])
        self.images = self.df['image_id'].values.tolist()

    def __getitem__(self, index):
        image_id = os.path.join(self.data_dir, 'test_images', self.images[index])
        # print('#####: ', image_id)

        image = cv2.imread(image_id, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image=image)['image']
            # image = self.transform(image)

        return image, image_id

    def __len__(self):
        return len(self.images)

    def read_test_dataset(self):
        path = os.path.join(self.data_dir, 'test')
        test_images = os.listdir(path)
        test_images.sort()

        return test_images


def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort=pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    # df = df.reset_index()
    df = df.drop('sort', axis=1)
    return df


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort(reverse=True)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def read_dataset(datadir, samples, class_to_idx):
    images = []
    labels = []

    for img in samples:
        image = os.path.join(datadir, img)
        images.append(image)
        labels.append(class_to_idx[img[:4]])

    tmp = list(zip(images, labels))
    random.shuffle(tmp)
    images, labels = zip(*tmp)

    return images, labels
