dataset_root = '/home/ace19/dl_data/shopee-product-matching'
dataset_name = 'product'
modelname = 'dm_nfnet_f0'
test_batch_size = 1
workers = 4
no_cuda = False
seed = 8

# cls_resume = 'experiments/shopee-product-matching/tf_efficientnet_b4_ns/' \
#              '(2021-03-17_21:10:32)product_fold3_380x380_tf_efficientnet_b4_ns_acc(54.97810)_loss(0.26047)_checkpoint30.pth.tar'

MODELS = [
    # 'experiments/shopee-product-matching/tf_efficientnet_b4_ns_1/(2021-03-17_21:10:32)product_fold0_380x380_tf_efficientnet_b4_ns_acc(54.07299)_loss(0.26073)_checkpoint29.pth.tar',
    # 'experiments/shopee-product-matching/tf_efficientnet_b4_ns_1/(2021-03-17_21:10:32)product_fold1_380x380_tf_efficientnet_b4_ns_acc(54.48175)_loss(0.26214)_checkpoint30.pth.tar',
    # 'experiments/shopee-product-matching/tf_efficientnet_b4_ns_1/(2021-03-17_21:10:32)product_fold2_380x380_tf_efficientnet_b4_ns_acc(54.36496)_loss(0.26438)_checkpoint29.pth.tar',
    # 'experiments/shopee-product-matching/tf_efficientnet_b4_ns_1/(2021-03-17_21:10:32)product_fold3_380x380_tf_efficientnet_b4_ns_acc(54.97810)_loss(0.26047)_checkpoint30.pth.tar',
    # 'experiments/shopee-product-matching/tf_efficientnet_b4_ns_1/(2021-03-17_21:10:32)product_fold4_380x380_tf_efficientnet_b4_ns_acc(54.46715)_loss(0.2576)_checkpoint30.pth.tar',
    'experiments/shopee-product-matching/dm_nfnet_f0/(2021-04-07_01:11:01)product_fold3_192x192_dm_nfnet_f0_acc(76.91533)_loss(8.66141)_checkpoint27.pth.tar'
]

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import Normalize

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

import timm
from pprint import pprint


class CosineSoftmaxModule(nn.Module):
    def __init__(self, features_dim, nclass=11014):
        super(CosineSoftmaxModule, self).__init__()
        self.nclass = nclass

        in_channels = features_dim  # BertModel: 768

        self.weights = torch.nn.Parameter(torch.randn(in_channels, self.nclass))
        self.scale = torch.nn.Parameter(F.softplus(torch.randn(())))
        self.fc = nn.Linear(in_channels, in_channels)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        # # self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-05)
        # self.features = nn.BatchNorm1d(in_channels, eps=1e-05)
        # self.flatten = Flatten()

        # # for arcface
        # self.in_features = self.pretrained.classifier.in_features
        # self.margin = ArcModule(in_features=in_channels, out_features=nclass)
        # self.bn1 = nn.BatchNorm2d(self.in_features)
        # # self.bn1 = nn.BatchNorm1d(in_channels, eps=1e-05)
        # self.dropout = nn.Dropout2d(0.2, inplace=True)
        # # self.dropout = nn.Dropout(p=0.4, inplace=True)
        # self.fc1 = nn.Linear(self.in_features * 12 * 12, in_channels)
        # # self.fc1 = nn.Linear(self.in_features, in_channels)
        # self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        ##################
        # COSINE-SOFTMAX
        ##################
        # x = x.view(-1, num_flat_features(x))
        # x = self.dropout(x)
        x = self.fc(x)

        features = x
        # Features in rows, normalize axis 1.
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = F.normalize(self.weights, p=2, dim=0, eps=1e-8)
        logits = self.scale.cuda() * torch.mm(features.cuda(), weights_normed.cuda())  # torch.matmul

        return features, logits

        # ##################
        # # ArcFace -
        # #   https://www.kaggle.com/underwearfitting/pytorch-densenet-arcface-validation-training
        # ##################
        # features = self.bn1(x)
        # features = self.dropout(features)
        # features = features.view(features.size(0), -1)
        # features = self.fc1(features)
        # features = self.bn2(features)
        # features = F.normalize(features)
        # if labels is not None:
        #     return self.margin(features, labels)
        #
        # return features


class Model(nn.Module):
    def __init__(self, backbone, nclass=11014):
        super(Model, self).__init__()
        self.backbone = backbone
        self.nclass = nclass

        model_names = timm.list_models(pretrained=True)
        pprint(model_names)
        self.pretrained = timm.create_model(self.backbone, pretrained=False, num_classes=nclass)
        # # Below code is used when if pretrained is False
        pre_model = torch.load('/home/ace19/.cache/torch/hub/checkpoints/dm_nfnet_f0-604f9c3a.pth')
        del pre_model['head.fc.weight']
        del pre_model['head.fc.bias']
        # self.pretrained.load_state_dict(pre_model, strict=False)

        self.in_channels = 512  # resnet18, resnet34
        if self.backbone in ['resnet18', 'resnet34', 'vgg16', 'vgg19']:
            self.in_channels = 512
        elif self.backbone in ['seresnext50_32x4d', 'resnext101_32x8d', 'resnext50_32x4d',
                               'resnest50d', 'resnest101e', 'resnest200e', 'resnet50',
                               'resnest269e', 'resnet101', 'resnet152', 'resnest50d_4s2x40d']:
            self.in_channels = 2048
        elif self.backbone.startswith('tf_efficientnet_b0'):
            self.in_channels = 1280
        elif self.backbone.startswith('tf_efficientnet_b1'):
            self.in_channels = 1280
        elif self.backbone.startswith('tf_efficientnet_b2'):
            self.in_channels = 1408
        elif self.backbone.startswith('tf_efficientnet_b3'):
            self.in_channels = 1536
        elif self.backbone.startswith('tf_efficientnet_b4'):
            self.in_channels = 1792
        elif self.backbone.startswith('tf_efficientnet_b5'):
            self.in_channels = 2048
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/nfnet.py
        elif self.backbone.startswith('dm_nfnet_f'):
            self.in_channels = 3072

        self.cosine_softmax = CosineSoftmaxModule(self.in_channels, nclass)

        # # TODO: make arcface func.
        # self.margin = ArcModule(in_features=self.in_channels, out_features=nclass)
        # # self.bn1 = nn.BatchNorm2d(self.in_channels)
        # # self.dropout = nn.Dropout2d(0.4, inplace=True)
        # # self.fc1 = nn.Linear(self.in_channels * 16 * 16, self.in_channels)    # original
        # self.fc1 = nn.Linear(self.in_channels, self.in_channels)
        # self.bn2 = nn.BatchNorm1d(self.in_channels)

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

        elif self.backbone.startswith('dm_nfnet'):
            x = self.pretrained.stem(x)
            x = self.pretrained.stages(x)
            x = self.pretrained.final_conv(x)
            x = self.pretrained.final_act(x)
            x = self.pretrained.head.global_pool(x)

        return self.cosine_softmax(x)


        # ##################
        # # ArcFace -
        # #   https://www.kaggle.com/underwearfitting/pytorch-densenet-arcface-validation-training
        # ##################
        # # features = self.bn1(x)
        # # features = self.dropout(features)
        # # features = features.view(features.size(0), -1)
        # features = self.fc1(x)
        # features = self.bn2(features)
        # features = F.normalize(features, eps=1e-8)
        # if labels is not None:
        #     return self.margin(features, labels)
        #
        # return features


import cv2
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


import os
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


# init the model
model = Model(backbone=modelname)
# model.half()  # to save space.
print('\n-------------- model details --------------')
print(model)

if cuda:
    model.cuda()
    model = nn.DataParallel(model)

test_df = pd.read_csv("/home/ace19/dl_data/shopee-product-matching/test.csv")

CHUNK = 1024 * 4
print('Computing image embeddings...')
CTS = len(test_df) // CHUNK

if len(test_df) % CHUNK != 0:
    CTS += 1

features_pool = []
for resume in MODELS:
    if resume is not None:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch'] + 1
            best_pred = checkpoint['best_pred']
            acc_lst_train = checkpoint['acc_lst_train']
            acc_lst_val = checkpoint['acc_lst_val']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no resume checkpoint found at '{}'".format(resume))

    model.eval()

    embeds = []
    # embed_posids = []
    for i, j in enumerate(range(CTS)):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b, len(test_df))
        print('chunk', a, 'to', b)

        galleryset = ProductTestDataset(data_dir=dataset_root,
                                        csv=test_df.iloc[a:b],
                                        transform=test_augmentation())

        gallery_loader = torch.utils.data.DataLoader(galleryset, batch_size=8, num_workers=workers)

        gallery_features_list = []
        #     gallery_path_list = []
        #         gallery_posid_list = []
        tbar = tqdm(gallery_loader, desc='\r')
        for batch_idx, (data, pos_id, img_path) in enumerate(tbar):
            if cuda:
                data = data.cuda()

            with torch.no_grad():
                features, output = model(data)

                gallery_features_list.append(features.cpu().numpy())
        #             gallery_path_list.extend(img_path)
        #                 gallery_posid_list.append(pos_id)

        embeds.extend(gallery_features_list)
    #     embed_posids.extend(gallery_posid_list)

    features_pool.append(np.concatenate(embeds))
    _ = gc.collect()

# -------------
# max pooling.
# -------------
gallery_features = features_pool[0]
for i in range(1, len(features_pool)):
    gallery_features = np.maximum(gallery_features, features_pool[i])

# gallery_posids = np.concatenate(embed_posids)
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
