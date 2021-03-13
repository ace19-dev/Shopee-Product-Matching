import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

np.random.seed(99)  # Reproducible random state

src = '/home/ace19/dl_data/shopee-product-matching'
train_df = pd.read_csv(os.path.join(src, 'train.csv'))

label_means = {}
label_stds = {}
for i in range(5):
    label = train_df.loc[train_df['label'] == i]
    label_lst = label['image_id'].tolist()

    means = []
    stds = []
    vals = []
    for idx, file in enumerate(tqdm(label_lst)):
        img_name = os.path.join(src, 'train_images', file)
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # Tuple이며 (width, height)
        # image = cv2.resize(image, (200, 150), interpolation=cv2.INTER_AREA)

        # val = np.reshape(image, -1)
        # vals.append(val)

        # val = np.reshape(image, -1)
        vals.append(image)

        # img_mean = np.mean(val) / 255
        # means.append(img_mean)
        #
        # img_std = np.std(val) / 255
        # stds.append(img_std)

    # label_means[i] = means
    # label_stds[i] = stds

    # 산점도 그리기
    # y = np.full_like(stds, i)
    # colors = np.full_like(stds, i)
    # area = 3
    # plt.scatter(stds, y, s=area, c=colors, alpha=0.5)
    # plt.show()

    # hist mean
    # m_vals = np.mean(vals, axis=0)
    # ax.hist(m_vals)
    # plt.savefig('cls_{}.jpg'.format(i))

    # imshow
    m_vals = np.mean(vals, axis=0)
    ax.imshow(m_vals)
    plt.imsave('cls_{}.jpg'.format(i), m_vals/255)
