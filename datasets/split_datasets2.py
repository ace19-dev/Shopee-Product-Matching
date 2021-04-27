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
import time
import random
import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder


def split_train_test(logger):
    shopee_product_df = pd.read_csv(os.path.join(args.source, 'shopee_product.csv'))
    logger.info('total: {}'.format(len(shopee_product_df)))
    logger.info('shopee_product shape: {}\n'.format(shopee_product_df.shape))
    # # delete unusual
    # shopee_product_df = shopee_product_df.loc[~shopee_product_df.filename.isin(
    #     ["64faf0b221af4767ba8c167b228fde00.jpg", "d946ee19ac1d2997bac5f18ce75656cb.jpg"])].reset_index(drop=True)

    # delete outliers
    # shopee_product_df = shopee_product_df[~shopee_product_df['image_id'].isin(OUTLIERS)]
    # logger.info('total:\n', len(shopee_product_df), '\n')
    # delete unusual
    # shopee_product_df = shopee_product_df[~shopee_product_df['image_id'].isin(UNUSUAL)]
    # logger.info('total:\n', len(shopee_product_df), '\n')

    logger.info('unique posting id: {}'.format(len(shopee_product_df['posting_id'].unique())))
    logger.info('unique image: {}'.format(len(shopee_product_df['image'].unique())))
    logger.info('unique image phash: {}'.format(len(shopee_product_df['image_phash'].unique())))
    logger.info('unique title: {}'.format(len(shopee_product_df['title'].unique())))
    logger.info('unique label group: {}'.format(len(shopee_product_df['label_group'].unique())))  # 11014
    logger.info('\n')
    logger.info('image counts:')
    logger.info(shopee_product_df['image'].value_counts())
    logger.info('\n\n')

    # https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175614
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    results = []
    for idx, (train_idx, test_idx) in enumerate(skf.split(shopee_product_df, shopee_product_df['label'])):
        train_fold = shopee_product_df.iloc[train_idx]
        test_fold = shopee_product_df.iloc[test_idx]

        logger.info('split: {} \n'.format(idx))
        logger.info('train_fold num: {}'.format(len(train_fold)))
        num = len(train_fold.groupby(by=['label'], as_index=False))
        logger.info('label num: {}, label/train_fold: {:.3f}'.format(
            num, num / len(train_fold)))

        logger.info('test_fold num: {}'.format(len(test_fold)))
        num = len(test_fold.groupby(by=['label'], as_index=False))
        logger.info('label num: {}, label/test_fold: {:.3f}'.format(
            num, num / len(test_fold)))

        logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        results.append((train_fold, test_fold))

    for i, splited in enumerate(results):
        train, test = splited
        # tmp_train = pd.DataFrame({'filename': train['filename'].tolist(), 'category': train['category'].tolist()})
        tmp_train = shopee_product_df.copy()
        tmp_train = tmp_train[tmp_train['posting_id'].isin(train['posting_id'].values)]
        tmp_train.to_csv(os.path.join(args.target, 'train%d_%d.csv' % (i, len(train))), index=False)
        np.save(args.target + '/test_fold%d_%d.npy' % (i, len(test)), test['posting_id'].tolist())

    '''
    each set contains approximately the same percentage of samples of each target class as the complete set.
    the test split has at least one y which has value 1
    
        [ 1  2  3  6  7  8  9 10 12 13 14 15] [ 0  4  5 11]
        [ 0  2  3  4  5  6  7 10 11 12 13 15] [ 1  8  9 14]
        [ 0  1  3  4  5  7  8  9 11 12 13 14] [ 2  6 10 15]
        [ 0  1  2  4  5  6  8  9 10 11 14 15] [ 3  7 12 13]
    '''


def split_train_val(logger):
    train_df = pd.read_csv(os.path.join(args.source, 'train3_27400.csv'))
    logger.info('total: {}'.format(len(train_df)))
    logger.info('train shape: {}\n'.format(train_df.shape))

    logger.info('unique posting id: {}'.format(len(train_df['posting_id'].unique())))
    logger.info('unique image: {}'.format(len(train_df['image'].unique())))
    logger.info('unique image phash: {}'.format(len(train_df['image_phash'].unique())))
    logger.info('unique title: {}'.format(len(train_df['title'].unique())))
    logger.info('unique label group: {}'.format(len(train_df['label_group'].unique())))  # 11014
    logger.info('unique label: {}'.format(len(train_df['label'].unique())))
    logger.info('\n')
    logger.info('image counts:')
    logger.info(train_df['image'].value_counts())
    logger.info('\n\n')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
    results = []
    for idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]

        logger.info('split: {} \n'.format(idx))
        logger.info('train_fold num: {}'.format(len(train_fold)))
        num = len(train_fold.groupby(by=['label'], as_index=False))
        logger.info('label num: {}, label/train_fold: {:.3f}'.format(
            num, num / len(train_fold)))

        logger.info('val_fold num: {}'.format(len(val_fold)))
        num = len(val_fold.groupby(by=['label'], as_index=False))
        logger.info('label num: {}, label/val_fold: {:.3f}'.format(
            num, num / len(val_fold)))

        logger.info('++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        results.append((train_fold, val_fold))

    for i, splited in enumerate(results):
        train, valid = splited
        np.save(args.target + '/train_fold%d_%d.npy' % (i, len(train)), train['posting_id'].tolist())
        np.save(args.target + '/valid_fold%d_%d.npy' % (i, len(valid)), valid['posting_id'].tolist())


# def test_stratifiedkfold():
#     X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
#     y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#     skf = StratifiedKFold(n_splits=5, shuffle=True)
#     for train_index, test_index in skf.split(X, y):
#         print("%s %s" % (train_index, test_index))


# # train/test 로 나눠질 때, 그룹에 속한 원소가 쪼개지지 않는다.
# def test_groupkfold():
#     X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
#     y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
#     groups = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd']
#     gkf = GroupKFold(n_splits=3)
#     for train_index, test_index in gkf.split(X, y, groups=groups):
#         print("%s %s" % (train_index, test_index))


def main(args):
    if args.target is not None:
        if not os.path.exists(args.target):
            os.makedirs(args.target)

    _time = time.strftime('%Y-%m-%d_%H:%M:%S')
    log_file = '({})split_train_val.log'.format(_time)
    final_log_file = args.target + '/' + log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    # https://www.kaggle.com/vishnurapps/undersanding-kfold-stratifiedkfold-and-groupkfold
    # test_groupkfold()

    # reference on https://www.kaggle.com/reighns/groupkfold-efficientbnet
    # split_train_test(logger)
    split_train_val(logger)

    # train_df = pd.read_csv(os.path.join(args.source, 'train.csv'))
    # logger.info('total train: {}'.format(len(train_df)))
    # logger.info('train shape: {}\n'.format(train_df.shape))
    # split_train_val(train_df, 'test', 6, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                        type=str,
                        default='/home/ace19/dl_data/shopee-product-matching/fold_text',
                        help='Where is train image to load')
    parser.add_argument('--target', type=str,
                        default='/home/ace19/dl_data/shopee-product-matching/fold_text',
                        help='directory to save splited dataset')

    args = parser.parse_args()
    main(args)
