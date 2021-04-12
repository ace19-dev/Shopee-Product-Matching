import math
from pprint import pprint

from transformers import BertModel, BertConfig, BertForSequenceClassification

import torch
import torch.nn as nn
from torch.nn import functional as F

from training.metric_learning_losses import *


class CosineSoftmaxModule(nn.Module):
    def __init__(self, features_dim, nclass=11014):
        super(CosineSoftmaxModule, self).__init__()
        self.nclass = nclass

        in_channels = features_dim  # BertModel: 768

        # self.fc_scale = 12 * 12  # effinet-b4
        self.weights = torch.nn.Parameter(torch.randn(in_channels, self.nclass))
        nn.init.xavier_uniform_(self.weights)
        self.scale = torch.nn.Parameter(F.softplus(torch.randn(())))
        # nn.init.xavier_uniform_(self.scale)
        self.fc = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(p=0.1, inplace=False)

    # pooler_output
    def forward(self, x):
        ##################
        # COSINE-SOFTMAX
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
    def __init__(self,
                 n_classes,
                 model_name='efficientnet_b0',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.1,
                 loss_module='arcface',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785,
                 pretrained=True):
        super(Model, self).__init__()
        self.model_name = model_name
        # self.nclass = nclass

        # TODO: use various model
        # self.pretrained = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased",
        #                                                       num_labels=nclass)
        config = BertConfig.from_pretrained('bert-base-multilingual-uncased')
        self.backbone = BertModel(config)
        # self.backbone = BertForSequenceClassification(config)
        # (pooler): BertPooler(
        #     (dense): Linear(in_features=768, out_features=768, bias=True)
        #     (activation): Tanh()
        # )
        # (dropout): Dropout(p=0.1, inplace=False)
        # (classifier): Linear(in_features=768, out_features=2, bias=True)
        print(self.backbone)

        final_in_features = self.backbone.pooler.dense.out_features

        self.cosine_softmax = CosineSoftmaxModule(final_in_features, n_classes)

        # # arcface
        # self.use_fc = use_fc
        # if use_fc:
        #     self.dropout = nn.Dropout(p=dropout)
        #     self.fc = nn.Linear(final_in_features, fc_dim)
        #     self.bn = nn.BatchNorm1d(fc_dim)
        #     self._init_params()
        #     final_in_features = fc_dim
        #
        # self.loss_module = loss_module
        # if loss_module == 'arcface':
        #     self.final = ArcMarginProduct(final_in_features, n_classes,
        #                                   s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        # elif loss_module == 'cosface':
        #     self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        # elif loss_module == 'adacos':
        #     self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        # else:
        #     self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    # cosine-softmax
    def forward(self, input_ids, input_mask):
        if self.model_name.startswith('bert'):
            outputs = self.backbone(input_ids=input_ids,
                                      attention_mask=input_mask,
                                      output_hidden_states=True)

        return self.cosine_softmax(outputs['pooler_output'])

    # # ArcFace
    # # https://www.kaggle.com/underwearfitting/pytorch-densenet-arcface-validation-training
    # def forward(self, input_ids, input_mask, labels):
    #     if self.model_name.startswith('bert'):
    #         outputs = self.backbone(input_ids=input_ids,
    #                                 attention_mask=input_mask, )
    #
    #     x = outputs['pooler_output']
    #     if self.use_fc:
    #         x = self.dropout(x)
    #         x = self.fc(x)
    #         feature = self.bn(x)
    #
    #     if self.loss_module in ('arcface', 'cosface', 'adacos'):
    #         logits = self.final(feature, labels)
    #     else:
    #         logits = self.final(feature)
    #     return feature, logits
