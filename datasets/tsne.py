'''
t-SNE stands for t-distributed stochastic neighbor embedding.
It is a technique for dimensionality reduction that is best suited for the visualization of high dimensional data-set.

WARNING!! need sampling to avoid big resource consumption
'''

import time
import os
import cv2
import argparse
import numpy as np
import seaborn as sns

import matplotlib.patheffects as PathEffects
from matplotlib import pyplot as plt

from sklearn.manifold import TSNE

# t-SNE is a randomized algorithm,
# i.e every time we run the algorithm it returns slightly different results on the same data-set.
# To control this we set a random state with some arbitrary value.
# Random state is used here to seed the cost function of the algorithm.
RS = 25111993


# An user defined function to create scatter plot of vectors
def scatter(x, colors, num_cls):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_cls))

    # We create a scatter plot.
    f = plt.figure(figsize=(48, 48))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=120, c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each cluster.
    txts = []
    for i in range(num_cls):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def crop_size(crop_ratio):
    weight = 2448
    height = 1000

    start_h = int(height * (crop_ratio/2))
    h = height - (start_h * 2)

    start_w = int(weight * crop_ratio)
    w = weight - (start_w * 2)

    return start_h, start_w, h, w


def main(args):
    if args.result is not None:
        if not os.path.exists(args.result):
            os.makedirs(args.result)

    # To make nice plots.
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 2.5})

    # start_h, start_w, h, w = crop_size(0.2)

    raws = []
    labels = []
    cls_lst = os.listdir(args.train_dataset_dir)
    cls_lst = sorted(cls_lst)
    num_cls = len(cls_lst)
    for idx, img in enumerate(cls_lst):
        cls_path = os.path.join(args.train_dataset_dir, img)
        image_lst = os.listdir(cls_path)
        for img in image_lst:
            img = cv2.imread(os.path.join(cls_path, img))
            # center crop
            # img = img[start_h:start_h+h, start_w:start_w+w]
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # resize to avoid time consuming
            img = cv2.resize(img, (400, 300))
            raws.append(img)
            labels.append(idx)

    raws = np.asarray(raws)
    tsne = TSNE(n_components=2, learning_rate=300.0, verbose=2, random_state=RS)
    # perform t-SNE embedding
    start = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'tSNE start: {start}')
    result = tsne.fit_transform(np.reshape(raws, (raws.shape[0], -1)))
    end = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'tSNE end: {end}')

    # sample_data1 = result[0:10, 0].tolist()
    # sample_data1 = pd.Series(sample_data1)
    # print('sample 1', sample_data1)
    # sample_data2 = result[0:10, 1].tolist()
    # sample_data2 = pd.Series(sample_data2)
    # print('sample 2', sample_data2)

    print('>>> plot the result\n')
    # plot 1
    # sns.palplot(np.array(sns.color_palette("hls", num_cls)))
    # scatter(result, np.array(labels), num_cls)
    # plt.savefig(os.path.join(args.result,
    #                          '%s_%s_tsne.png' % (time.strftime("%Y-%m-%d_%H:%M:%S"),
    #                                              ds)), dpi=120)

    # plot 2
    palette = np.array(sns.color_palette("hls", num_cls))
    f = plt.figure(figsize=(48, 48))
    plt.scatter(result[:, 0], result[:, 1], c=palette[np.asarray(labels)], cmap=plt.cm.get_cmap("jet", num_cls))
    plt.colorbar(ticks=range(num_cls))
    plt.savefig(os.path.join(args.result,
                             '%s_%s_tsne.png' % (time.strftime("%Y-%m-%d_%H:%M:%S"), 'train')), dpi=120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_dataset_dir',
        type=str,
        default='/home/ace19/dl_data/shopee-product-matching/train',
        help='Where is train image to load'
    )
    parser.add_argument(
        '--validation_dataset_dir',
        type=str,
        default='/home/ace19/dl_data/shopee-product-matching/validation',
        help='Where is validation image to load'
    )
    parser.add_argument(
        '--test_dataset_dir',
        type=str,
        default='/home/ace19/dl_data/shopee-product-matching/test',
        help='Where is test image to load'
    )
    parser.add_argument('--result', type=str,
                        default='result',
                        help='directory to save tSNE result')

    args = parser.parse_args()
    main(args)