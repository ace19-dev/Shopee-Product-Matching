import math
from pprint import pprint

from transformers import BertModel, BertConfig

import torch
import torch.nn as nn
from torch.nn import functional as F


class View(nn.Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """

    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Normalize(nn.Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """

    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_feature = 1
    for s in size:
        num_feature *= s

    return num_feature


class ArcModule(nn.Module):
    # margin = 0.5 # 0 for faster convergence, larger may be beneficial
    def __init__(self, in_features, out_features, s=10, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # print(type(cos_th), type(self.th), type(cos_th_m), type(self.mm))
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


class CosineSoftmaxModule(nn.Module):
    def __init__(self, features_dim, nclass=11014):
        super(CosineSoftmaxModule, self).__init__()
        self.nclass = nclass

        in_channels = features_dim  # BertModel: 768

        # self.fc_scale = 12 * 12  # effinet-b4
        self.weights = torch.nn.Parameter(torch.randn(in_channels, self.nclass))
        self.scale = torch.nn.Parameter(F.softplus(torch.randn(())))
        self.fc = nn.Linear(in_channels, in_channels)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        # # self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-05)
        # self.features = nn.BatchNorm1d(in_channels, eps=1e-05)
        # self.flatten = Flatten()
        # nn.init.constant_(self.features.weight, 1.0)
        # self.features.weight.requires_grad = False

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

    # pooler_output
    def forward(self, x):
        ##################
        # COSINE-SOFTMAX
        ##################
        # x = x.view(-1, num_flat_features(x))
        # x = F.dropout2d(x, p=0.2)
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

        # TODO: use various model
        # self.pretrained = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
        #                                                       num_labels=nclass)
        config = BertConfig.from_pretrained('bert-base-multilingual-cased')
        self.pretrained = BertModel(config)
        # print(self.pretrained)

        in_channels = 768  # BertForSequenceClassification
        self.cosine_softmax = CosineSoftmaxModule(in_channels, nclass)

    def forward(self, input_ids, input_mask):
        if self.backbone.startswith('Bert'):
            outputs = self.pretrained(input_ids=input_ids,
                                      attention_mask=input_mask,)

        return self.cosine_softmax(outputs['pooler_output'])