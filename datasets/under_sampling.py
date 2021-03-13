import os
import random

import numpy as np
import pandas as pd

np.random.seed(99)  # Reproducible random state

src = '/home/ace19/dl_data/shopee-product-matching'
train_df = pd.read_csv(os.path.join(src, 'train_2020.csv'))

# read major class
majority = train_df.loc[train_df['label'] == 3].index.values.tolist()
sample = random.sample(majority, 2500)
non_sample = list(set(majority) - set(sample))
filtering = train_df.drop(non_sample)
filtering.to_csv(os.path.join(src, 'train_undersample.csv'), index=False)
