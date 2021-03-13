import os
import random
import pandas as pd

root = '/home/ace19/dl_data/shopee-product-matching'

train_2020_df = pd.read_csv(os.path.join(root, 'train_2020.csv'))

images_all = []
labels = []

train_2018 = os.listdir(os.path.join(root, 'train_2018'))
for idx, cls in enumerate(sorted(train_2018)):
    cls_path = os.path.join(root, 'train_2018', cls)
    cls_images = os.listdir(cls_path)
    for img in cls_images:
        images_all.append(img)
        labels.append(idx)

train_2018_df = pd.DataFrame({'image_id': images_all, 'label': labels})
train_2018_df.to_csv(os.path.join(root, 'train_2018.csv'), index=False)

# add unlabeled data
train_2018_unlabeled_image = os.listdir(os.path.join(root, 'unlabeled_2018'))
train_2018_unlabeled_label = []
for _ in train_2018_unlabeled_image:
    train_2018_unlabeled_label.append(random.randint(0, 4))

# train_2018_unlabeled_df = pd.DataFrame({'image_id': train_2018_unlabeled_image,
#                                         'label': train_2018_unlabeled_label})
# train = pd.concat([train_2020_df, train_2018_df, train_2018_unlabeled_df])
train = pd.concat([train_2020_df, train_2018_df])

# train = pd.concat([train_2020_df, train_2018_df])
train.to_csv(os.path.join(root, 'train.csv'), index=False)
