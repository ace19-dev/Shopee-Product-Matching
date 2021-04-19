import math
from pprint import pprint

import timm
import torch
import torch.nn as nn
from torch.nn import functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_feature = 1
    for s in size:
        num_feature *= s

    return num_feature


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class CosineSoftmaxModule(nn.Module):
    def __init__(self, features_dim, nclass=11014):
        super(CosineSoftmaxModule, self).__init__()
        self.nclass = nclass

        in_channels = features_dim

        self.weights = torch.nn.Parameter(torch.randn(in_channels, self.nclass))
        nn.init.xavier_uniform_(self.weights)
        self.scale = torch.nn.Parameter(F.softplus(torch.randn(())))
        # nn.init.xavier_uniform_(self.scale)
        self.fc = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(p=0.2, inplace=False)

    def forward(self, x):
        ##################
        # cosine-softmax
        ##################
        # x = x.view(-1, num_flat_features(x))
        x = self.dropout(x)
        x = self.fc(x)

        features = x
        # Features in rows, normalize axis 1.
        features = F.normalize(features, p=2, dim=1, eps=1e-8)
        # Mean vectors in colums, normalize axis 0.
        weights_normed = F.normalize(self.weights, p=2, dim=0, eps=1e-8)
        logits = self.scale.cuda() * torch.mm(features.cuda(), weights_normed.cuda())  # torch.matmul

        return features, logits


class Model(nn.Module):
    def __init__(self, model_name, use_fc=False, fc_dim=512, nclass=11014):
        super(Model, self).__init__()
        self.model_name = model_name
        self.use_fc = use_fc
        self.nclass = nclass

        print('Building Model Backbone for {} model'.format(model_name))
        model_names = timm.list_models(pretrained=True)
        pprint(model_names)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=nclass)
        # # Below code is used when if pretrained is False
        # https://www.kaggle.com/parthdhameliya77/pytorch-resnext50-32x4d-image-tfidf-inference
        # pre_model = torch.load('/home/ace19/.cache/torch/hub/checkpoints/dm_nfnet_f0-604f9c3a.pth')
        # del pre_model['fc.weight']
        # del pre_model['fc.bias']
        # self.backbone.load_state_dict(pre_model, strict=False)

        self.in_channels = 512  # resnet18, resnet34
        if self.model_name in ['resnet18', 'resnet34', 'vgg16', 'vgg19']:
            self.in_channels = 512
        elif self.model_name in ['seresnext50_32x4d', 'resnext101_32x8d', 'resnext50_32x4d',
                                 'resnest50d', 'resnest101e', 'resnest200e', 'resnet50',
                                 'resnest269e', 'resnet101', 'resnet152', 'resnest50d_4s2x40d']:
            self.in_channels = 2048
        elif self.model_name.startswith('tf_efficientnet_b0'):
            self.in_channels = 1280
        elif self.model_name.startswith('tf_efficientnet_b1'):
            self.in_channels = 1280
        elif self.model_name.startswith('tf_efficientnet_b2'):
            self.in_channels = 1408
        elif self.model_name.startswith('tf_efficientnet_b3'):
            self.in_channels = 1536
        elif self.model_name.startswith('tf_efficientnet_b4'):
            self.in_channels = 1792
        elif self.model_name.startswith('tf_efficientnet_b5'):
            self.in_channels = 2048
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/nfnet.py
        elif self.model_name.startswith('dm_nfnet_f'):
            self.in_channels = 3072

        ##################
        # cosine-softmax
        ##################
        self.cosine_softmax = CosineSoftmaxModule(self.in_channels, nclass)

        # ##################
        # # ArcFace - https://www.kaggle.com/parthdhameliya77/pytorch-resnext50-32x4d-image-tfidf-inference
        # ##################
        # self.margin = ArcMarginProduct(fc_dim, self.nclass)
        # self.pooling = nn.AdaptiveAvgPool2d(1)
        # self.dropout = nn.Dropout(p=0.1, inplace=False)
        # self.fc = nn.Linear(self.in_channels, fc_dim)
        # self.bn = nn.BatchNorm1d(fc_dim)
        # nn.init.xavier_normal_(self.fc.weight)
        # nn.init.constant_(self.fc.bias, 0)
        # nn.init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, labels):
        batch_size = x.shape[0]

        if self.model_name.startswith('tf_efficientnet'):
            x = self.backbone.conv_stem(x)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)
            x = self.backbone.blocks(x)
            x = self.backbone.conv_head(x)
            x = self.backbone.bn2(x)
            x = self.backbone.act2(x)
            x = self.backbone.global_pool(x)

        elif self.model_name.startswith('resnet') or \
                self.model_name.startswith('resnext') or \
                self.model_name.startswith('seresnext') or \
                self.model_name.startswith('resnest'):
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
            x = self.backbone.global_pool(x)

        elif self.model_name.startswith('dm_nfnet'):
            x = self.backbone.stem(x)
            x = self.backbone.stages(x)
            x = self.backbone.final_conv(x)
            x = self.backbone.final_act(x)
            x = self.backbone.head.global_pool(x)

        # x = self.pooling(x).view(batch_size, -1)

        ##################
        # cosine-softmax
        ##################
        return self.cosine_softmax(x)

        ##################
        # ArcFace - https://www.kaggle.com/parthdhameliya77/pytorch-resnext50-32x4d-image-tfidf-inference
        ##################
        # ---------------
        # original paper
        # ---------------
        # features = self.backbone.features(x)
        # features = self.bn1(features)
        # features = self.dropout(features)
        # features = features.view(features.size(0), -1)
        # features = self.fc1(features)
        # features = self.bn2(features)
        # features = F.normalize(features)
        # return self.margin(features, labels)

        # if self.use_fc:
        #     x = self.dropout(x)
        #     x = self.fc(x)
        #     x = self.bn(x)
        #
        # # logits when training
        # return self.margin(x, labels)
        #
        # # features when inference
        # # return x
