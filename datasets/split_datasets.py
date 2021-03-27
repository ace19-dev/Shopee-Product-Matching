'''
 For example, in the splitting process, there are 15 images of patient 1 in the training set,
 and there are 5 images in the validation set; then this may not be ideal since the model has already seen 15
 of the images for patient 1 and can easily remember features that are unique to patient 1, and therefore predict
 well in the validation set for the same patient 1. Therefore, this cross validation method may give
 over optimistic results and fail to generalize well to more unseen images.

 reference on https://www.kaggle.com/reighns/groupkfold-efficientbnet

'''

import argparse
import glob
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.utils import shuffle


# TODO: modify
def create_label_column():
    train_df = pd.read_csv(os.path.join(args.source, 'train_ori.csv'))

    # https://www.kaggle.com/vicioussong/rapids-tfidfvectorizer-cv-0-734
    label_group_to_label = train_df.groupby('label_group').posting_id.agg('unique').to_dict()
    for lbl, (key, value) in enumerate(sorted(label_group_to_label.items())):
        idxs = train_df.loc[train_df['posting_id'].isin(value)].index.tolist()
        for idx in idxs:
            train_df.loc[idx, 'label'] = lbl

    train_df.to_csv(os.path.join(args.source, 'train.csv'), index=False)


def split_train_val3():
    # NUM_CLASSES = 5

    train_df = pd.read_csv(os.path.join(args.source, 'train.csv'))
    print('total: ', len(train_df))
    print('train shape: ', train_df.shape, '\n')
    # # delete old unusual
    # train_df = train_df.loc[~train_df.filename.isin(
    #     ["64faf0b221af4767ba8c167b228fde00.jpg", "d946ee19ac1d2997bac5f18ce75656cb.jpg"])].reset_index(drop=True)

    # delete outliers
    # train_df = train_df[~train_df['image_id'].isin(OUTLIERS)]
    # print('total:\n', len(train_df), '\n')
    # delete unusual
    # train_df = train_df[~train_df['image_id'].isin(UNUSUAL)]
    # print('total:\n', len(train_df), '\n')

    print('unique posting id: ', len(train_df['posting_id'].unique()))
    print('unique image: ', len(train_df['image'].unique()))
    print('unique image phash: ', len(train_df['image_phash'].unique()))
    print('unique title: ', len(train_df['title'].unique()))
    print('unique label group: ', len(train_df['label_group'].unique()))  # 11014
    print(train_df['image'].value_counts())
    # print(train_df['label_group'].astype(int))

    # https://stackoverflow.com/questions/50375985/pandas-add-column-with-value-based-on-condition-based-on-other-columns
    # train_df['label2'] = np.where(train_df['label'].isin([0, 1, 2, 3]), 1, 0)
    # train_df.to_csv(os.path.join(args.source, 'train2.csv'), sep=',', na_rep='NaN')

    # # https://www.kaggle.com/underwearfitting/pytorch-densenet-arcface-validation-training
    # gkf = GroupKFold(n_splits=5)
    # df_train['fold'] = -1
    # for fold, (train_idx, valid_idx) in enumerate(gkf.split(df_train, None, df_train.label_group)):
    #     df_train.loc[valid_idx, 'fold'] = fold

    # https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175614
    # sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_seed=42)
    gkf = GroupKFold(n_splits=5)
    x_shuffled, y_shuffled, groups_shuffled = \
        shuffle(train_df, train_df['label'], train_df['image'].tolist(), random_state=8)
    results = []
    for idx, (train_idx, val_idx) in enumerate(gkf.split(x_shuffled, y_shuffled, groups=groups_shuffled)):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]

        print('\nsplit: ', idx)
        # train = train_fold.groupby(by=['label'], as_index=False).sum()
        # print('train_df:\n', train_fold.groupby(by=['label'], as_index=False).sum())
        print('train_fold num: ', len(train_fold))
        num = len(train_fold.groupby(by=['label'], as_index=False))
        print('label num: {}, \nlabel/train_fold: {:.3f}'.format(
            num, num / len(train_fold)))

        print('\nval_fold num: ', len(val_fold))
        num = len(val_fold.groupby(by=['label'], as_index=False))
        print('label num: {}, \nlabel/val_fold: {:.3f}'.format(
            num, num / len(val_fold)))

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        results.append((train_fold, val_fold))

    for i, splited in enumerate(results):
        train, valid = splited
        np.save(args.target + '/train_fold%d_%d.npy' % (i, len(train)), train['posting_id'].tolist())
        np.save(args.target + '/valid_fold%d_%d.npy' % (i, len(valid)), valid['posting_id'].tolist())

    '''
    each set contains approximately the same percentage of samples of each target class as the complete set.
    the test split has at least one y which has value 1
    
        [ 1  2  3  6  7  8  9 10 12 13 14 15] [ 0  4  5 11]
        [ 0  2  3  4  5  6  7 10 11 12 13 15] [ 1  8  9 14]
        [ 0  1  3  4  5  7  8  9 11 12 13 14] [ 2  6 10 15]
        [ 0  1  2  4  5  6  8  9 10 11 14 15] [ 3  7 12 13]
    '''


def make_train_npy():
    train_df = pd.read_csv(os.path.join(args.source, 'train.csv'))
    print('total:\n', len(train_df), '\n')

    image_id = train_df['image_id'].tolist()
    np.save(args.target + '/train_all_%d.npy' % (len(image_id)), image_id)


def test_stratifiedkfold():
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    skf = StratifiedKFold(n_splits=4, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        print("%s %s" % (train_index, test_index))


def test_groupkfold():
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    groups = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd']
    gkf = GroupKFold(n_splits=3)
    for train_index, test_index in gkf.split(X, y, groups=groups):
        print("%s %s" % (train_index, test_index))


def main(args):
    if args.target is not None:
        if not os.path.exists(args.target):
            os.makedirs(args.target)

    # https://www.kaggle.com/vishnurapps/undersanding-kfold-stratifiedkfold-and-groupkfold
    # test_groupkfold()

    # reference on https://www.kaggle.com/reighns/groupkfold-efficientbnet
    # create_label_column()
    split_train_val3()

    # temp
    # make_train_npy()

    # reference on https://www.kaggle.com/shonenkov/merge-external-data
    # split_train_val2()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        type=str,
                        default='/home/ace19/dl_data/shopee-product-matching',
                        help='Where is train image to load')
    parser.add_argument('--target', type=str,
                        default='/home/ace19/dl_data/shopee-product-matching/fold',
                        help='directory to save splited dataset')

    args = parser.parse_args()
    main(args)
