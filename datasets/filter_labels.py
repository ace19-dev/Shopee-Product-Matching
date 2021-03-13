import os

import numpy as np
import pandas as pd

np.random.seed(99)  # Reproducible random state

src = '/home/ace19/dl_data/shopee-product-matching'
train_df = pd.read_csv(os.path.join(src, 'train.csv'))

# Drop label 4 images
label = train_df.loc[train_df['label'] == 4].index.values.tolist()
filtering = train_df.drop(label)
filtering.to_csv(os.path.join(src, 'train_cls4.csv'), index=False)
