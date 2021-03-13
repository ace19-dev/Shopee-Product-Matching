import os

import numpy as np
import pandas as pd

np.random.seed(99)  # Reproducible random state

src = '/home/ace19/dl_data/shopee-product-matching'
train_df = pd.read_csv(os.path.join(src, 'train.csv'))

# train_df.loc[train_df['label'].isin(['0', '1', '2', '3']), 'label2'] = '1'
# train_df.loc[train_df['label'] == '4', 'label2'] = '0'
# train_df.to_csv(os.path.join(src, 'train_cls2.csv'), index=False)

# split classes partly
# train4_df = train_df.loc[train_df['label'].isin(['0', '1', '2', '3'])]
# train4_df.to_csv(os.path.join(src, 'train_cls4.csv'), index=False)

# https://m.blog.naver.com/PostView.nhn?blogId=nomadgee&logNo=220812476823&proxyReferer=https:%2F%2Fwww.google.com%2F
train_cls0_df = train_df.loc[train_df['label'] == 0]
cls0_sample_df = train_cls0_df.sample(frac=0.6)

train_cls1_df = train_df.loc[train_df['label'] == 1]
cls1_sample_df = train_cls1_df.sample(frac=0.6)

train_cls2_df = train_df.loc[train_df['label'] == 2]
cls2_sample_df = train_cls2_df.sample(frac=0.6)

train_cls3_df = train_df.loc[train_df['label'] == 3]
cls3_sample_df = train_cls3_df.sample(frac=0.6)

train_cls4_df = train_df.loc[train_df['label'] == 4]
cls4_sample_df = train_cls4_df.sample(frac=0.6)

train = pd.concat([cls0_sample_df, cls1_sample_df, cls2_sample_df,
                   cls3_sample_df, cls4_sample_df])
# shuffle
train = train.sample(frac=1).reset_index(drop=True)

train.to_csv(os.path.join(src, 'train_half.csv'), index=False)
