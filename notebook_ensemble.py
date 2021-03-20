import argparse
import os

dataset_root = '/home/ace19/dl_data/shopee-product-matching'
dataset_name = 'product'
modelname = 'tf_efficientnet_b4_ns'
test_batch_size = 1
workers = 4
no_cuda = False
seed = 8

cls_resume = 'experiments/shopee-product-matching/tf_efficientnet_b4_ns/' \
             '(2021-03-17_21:10:32)product_fold3_380x380_tf_efficientnet_b4_ns_acc(54.97810)_loss(0.26047)_checkpoint30.pth.tar'

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.transforms import TenCrop, FiveCrop, ToTensor, Lambda, Normalize

import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations import SmallestMaxSize, HorizontalFlip, Compose, RandomCrop
from albumentations.pytorch import ToTensorV2

from timm.data.transforms_factory import transforms_imagenet_train
from timm.data.random_erasing import RandomErasing

# imagenet
normalize = Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])

CROP_HEIGHT = 380
CROP_WIDTH = 380


def test_augmentation():
    test_transform = [
        A.Resize(CROP_HEIGHT, CROP_WIDTH),
        A.Normalize(),
        ToTensorV2(),
    ]
    return A.Compose(test_transform)


import torch.nn as nn

import torch
from torch.nn import functional as F
import torchvision.models as torch_models

import timm
from pprint import pprint


class Model(nn.Module):
    def __init__(self, backbone, nclass=11014):
        super(Model, self).__init__()
        self.backbone = backbone
        self.ncode = 32
        #         self.nclass = nclass

        model_names = timm.list_models(pretrained=True)
        pprint(model_names)
        self.pretrained = timm.create_model(self.backbone, pretrained=False, num_classes=nclass)
        # Below code is used when if pretrained is False
        pre_model = torch.load('/home/ace19/.cache/torch/checkpoints/tf_efficientnet_b4_ns-d6313a46.pth')
        #         del pre_model['fc.weight']
        #         del pre_model['fc.bias']
        del pre_model['classifier.weight']
        del pre_model['classifier.bias']
        self.pretrained.load_state_dict(pre_model, strict=False)
        #         self.pretrained.fc = nn.Linear(in_channels, nclass),

        in_channels = 512  # resnet18, resnet34
        if self.backbone in ['resnet18', 'resnet34', 'vgg16', 'vgg19']:
            in_channels = 512
        elif self.backbone in ['seresnext50_32x4d', 'resnext101_32x8d', 'resnext50_32x4d',
                               'resnest50d', 'resnest101e', 'resnest200e', 'resnet50',
                               'resnest269e', 'resnet101', 'resnet152']:
            in_channels = 2048
        elif self.backbone.startswith('tf_efficientnet_b0'):
            in_channels = 1280
        elif self.backbone.startswith('tf_efficientnet_b1'):
            in_channels = 1280
        elif self.backbone.startswith('tf_efficientnet_b2'):
            in_channels = 1408
        elif self.backbone.startswith('tf_efficientnet_b3'):
            in_channels = 1536
        elif self.backbone.startswith('tf_efficientnet_b4'):
            in_channels = 1792
        elif self.backbone.startswith('tf_efficientnet_b5'):
            in_channels = 2048

        self.weights = torch.nn.Parameter(torch.randn(in_channels, nclass))
        self.scale = torch.nn.Parameter(F.softplus(torch.randn(())))
        self.fc = nn.Linear(in_channels, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        if self.backbone.startswith('tf_efficientnet'):
            x = self.pretrained.conv_stem(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.act1(x)
            x = self.pretrained.blocks(x)
            x = self.pretrained.conv_head(x)
            x = self.pretrained.bn2(x)
            x = self.pretrained.act2(x)
            x = self.pretrained.global_pool(x)
            # return self.pretrained.classifier(x)
        elif self.backbone.startswith('resnet') or \
                self.backbone.startswith('resnext') or \
                self.backbone.startswith('seresnext') or \
                self.backbone.startswith('resnest'):
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.act1(x)
            x = self.pretrained.maxpool(x)
            x = self.pretrained.layer1(x)
            x = self.pretrained.layer2(x)
            x = self.pretrained.layer3(x)
            x = self.pretrained.layer4(x)
            x = self.pretrained.global_pool(x)

        # COSINE-SOFTMAX
        # feature_dim = x.size()[1]
        # x = x.view(-1, num_flat_features(x))
        x = F.dropout2d(x, p=0.1)
        x = self.fc(x)

        features = x
        # Features in rows, normalize axis 1.
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = F.normalize(self.weights, p=2, dim=0, eps=1e-8)
        logits = self.scale.cuda() * torch.mm(features.cuda(), weights_normed.cuda())  # torch.matmul

        return features, logits


import cv2
import numpy as np
import torch.utils.data as data


class ProductTestDataset(data.Dataset):
    def __init__(self, data_dir, csv, transform=None):
        self.data_dir = data_dir
        self.csv = csv
        self.transform = transform

        #         self.df = pd.concat([pd.read_csv(data_dir + '/%s' % f) for f in self.csv])
        df = csv
        self.images = df['image'].values.tolist()
        # self.labels = df['label'].values.tolist()
        self.posting_id = df['posting_id'].values.tolist()

    def __getitem__(self, index):
        # TODO: will fix for test_images
        image_path = os.path.join(self.data_dir, 'test_images', self.images[index])
        # print('#####: ', image_path)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        # return image, self.posting_id[index], image_path, self.labels[index],
        return image, self.posting_id[index], image_path

    def __len__(self):
        return len(self.images)


import numpy as np

import torch.nn.functional as F


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


# TODO: modify
def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    # a = a.numpy()
    # b = b.numpy()

    if not data_is_normalized:
        # To avoid RuntimeWarning: invalid value encountered in true_divide import numpy as np
        # a_normed = F.normalize(a, p=2, dim=1, eps=1e-8)
        a_normed = np.linalg.norm(a, axis=1, keepdims=True)
        a = np.asarray(a) / np.where(a_normed == 0, 1, a_normed)
        # b_normed = F.normalize(a, p=2, dim=1, eps=1e-8)
        b_normed = np.linalg.norm(b, axis=1, keepdims=True)
        b = np.asarray(b) / np.where(b_normed == 0, 1, b_normed)
    else:
        a = np.asarray(a)
        b = np.asarray(b)

    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of M row-vectors (query points).
    y : ndarray
        A matrix of N row-vectors (gallery points).

    Returns
    -------
    # ndarray
    #     A vector of length M that contains for each entry in `y` the
    #     smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    return distances


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, metric, matching_threshold=None, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget  # Gating threshold for cosine distance
        self.samples = {}

    def distance(self, queries, galleries):
        """Compute distance between galleries and queries.

        Parameters
        ----------
        queries : ndarray
            An LxM matrix of L features of dimensionality M to match the given `galleries` against.
        galleries : ndarray
            An NxM matrix of N features of dimensionality M.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape LxN

        """
        return self._metric(queries, galleries)


def _print_distances(distance_matrix, top_n_indice):
    distances = []
    num_row, num_col = top_n_indice.shape
    for r in range(num_row):
        col = []
        for c in range(num_col):
            col.append(distance_matrix[r, top_n_indice[r, c]])
        distances.append(col)

    return distances


# TODO: 특정 거리를 설정해서 top 50 top
def match_n(top_n, galleries, queries):
    # The distance metric used for measurement to query.
    metric = NearestNeighborDistanceMetric("cosine")
    start = time.time()
    distance_matrix = metric.distance(queries, galleries)
    end = time.time()
    print("distance measure time: {}".format(end - start))

    # top_indice = np.argmin(distance_matrix, axis=1)
    # top_n_indice = np.argpartition(distance_matrix, top_n, axis=1)[:, :top_n]
    # top_n_dist = _print_distances(distance_matrix, top_n_indice)
    # top_n_indice2 = np.argsort(top_n_dist, axis=1)
    # dist2 = _print_distances(distance_matrix, top_n_indice2)

    # TODO: need improvement.
    top_n_indice = np.argsort(distance_matrix, axis=1)[:, :top_n]
    top_n_distance = _print_distances(distance_matrix, top_n_indice)

    return top_n_indice, top_n_distance


def show_retrieval_result(top_n_indice, top_n_distance,
                          gallery_posid_list, query_posid_list):
    #     submit_df = pd.read_csv("../input/shopee-product-matching/sample_submission.csv")
    #     submit_images = submit_df['posting_id'].values.tolist()

    query_posids = []
    gallery_posids = []

    col = top_n_indice.shape[1]
    for row_idx, query_posid in enumerate(query_posid_list):
        # query_posids.append(query_posid_list[row_idx])

        posids = []
        for i in range(col):
            # TODO: fix
            if top_n_distance[row_idx][i] < 1.0:
                continue

            posids.append(gallery_posid_list[top_n_indice[row_idx, i]])

        gallery_posids.append(' '.join(posids))

    return gallery_posids


import csv
import os
import time
import random
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASS = 11014
TOP_N = 50

# global variable
best_pred = 0.0
acc_lst_train = []
acc_lst_val = []

cuda = not no_cuda and torch.cuda.is_available()
print(cuda)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# galleryset = ProductTestDataset(data_dir=dataset_root,
#                                     csv=['test.csv'],
#                                     transform=test_augmentation())
# queryset = ProductTestDataset(data_dir=dataset_root,
#                               csv=['test.csv'],
#                               transform=test_augmentation())

# gallery_loader = torch.utils.data.DataLoader(galleryset, batch_size=8, num_workers=workers)
# query_loader = torch.utils.data.DataLoader(queryset, batch_size=1, num_workers=workers)


# init the model
model = Model(backbone=modelname)
# model.half()  # to save space.
print('\n-------------- model details --------------')
print(model)

if cuda:
    model.cuda()
    model = nn.DataParallel(model)

if cls_resume is not None:
    if os.path.isfile(cls_resume):
        print("=> loading checkpoint '{}'".format(cls_resume))
        checkpoint = torch.load(cls_resume)
        start_epoch = checkpoint['epoch'] + 1
        best_pred = checkpoint['best_pred']
        acc_lst_train = checkpoint['acc_lst_train']
        acc_lst_val = checkpoint['acc_lst_val']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(cls_resume, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no cls_resume checkpoint found at '{}'".format(cls_resume))
else:
    raise RuntimeError("=> config \'cls_resume\' is '{}'".format(cls_resume))

# gallery_features_list = []
gallery_path_list = []
gallery_posid_list = []
query_features_list = []
query_path_list = []
query_posid_list = []

model.eval()

test_df = pd.read_csv("/home/ace19/dl_data/shopee-product-matching/test.csv")

embeds = []
embed_posids = []

CHUNK = 1024 * 4
print('Computing image embeddings...')
CTS = len(test_df) // CHUNK

if len(test_df) % CHUNK != 0:
    CTS += 1

for i, j in enumerate(range(CTS)):

    a = j * CHUNK
    b = (j + 1) * CHUNK
    b = min(b, len(test_df))
    print('chunk', a, 'to', b)

    galleryset = ProductTestDataset(data_dir=dataset_root,
                                    csv=test_df.iloc[a:b],
                                    transform=test_augmentation())

    gallery_loader = torch.utils.data.DataLoader(galleryset, batch_size=test_batch_size,
                                                 num_workers=workers)

    gallery_features_list = []
    gallery_path_list = []
    gallery_posid_list = []
    tbar = tqdm(gallery_loader, desc='\r')
    for batch_idx, (data, pos_id, img_path) in enumerate(tbar):
        if cuda:
            data = data.cuda()

        with torch.no_grad():
            features, output = model(data)

            gallery_features_list.append(features.cpu().numpy())
            gallery_path_list.append(img_path)
            gallery_posid_list.append(pos_id)

    embeds.extend(gallery_features_list)
    embed_posids.extend(gallery_posid_list)

# print("\n ==> Copy query ... ")
# query_features_list = gallery_features_list.copy()
# query_path_list = gallery_path_list.copy()
# query_posid_list = gallery_posid_list.copy()

del model
_ = gc.collect()
gallery_features = np.concatenate(embeds)
gallery_posids = np.concatenate(embed_posids)
print('image embeddings shape', gallery_features.shape)

# TODO
# import cudf, cuml
import cupy

image_embeddings = cupy.array(gallery_features)

preds = []
print('Finding similar images...')

CTS = len(gallery_features) // CHUNK
if len(gallery_features) % CHUNK != 0:
    CTS += 1

for j in range(CTS):
    a = j * CHUNK
    b = (j + 1) * CHUNK
    b = min(b, len(gallery_features))
    print('chunk', a, 'to', b)

    cts = cupy.matmul(image_embeddings, image_embeddings[a:b].T).T

    for k in range(b - a):
        #         print(sorted(cts[k,], reverse=True))
        IDX = cupy.where(cts[k,] > 0.5)[0]
        o = test_df.iloc[cupy.asnumpy(IDX)].posting_id.values
        preds.append(' '.join(o.tolist()))

submit_df = pd.read_csv("/home/ace19/dl_data/shopee-product-matching/sample_submission.csv")

submit_df['matches'] = preds
submit_df[['posting_id', 'matches']].to_csv('submission.csv', index=False)
