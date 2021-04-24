import math
from pprint import pprint

# from transformers import BertModel, BertConfig, BertForSequenceClassification

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
    def __init__(self, bert_model, num_classes=11014, last_hidden_size=768):
        super(Model, self).__init__()
        self.bert_model = bert_model
        self.arc_margin = ArcMarginProduct(last_hidden_size,
                                           num_classes,
                                           s=30.0,
                                           m=0.50,
                                           easy_margin=False)

    def get_bert_features(self, batch):
        output = self.bert_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        last_hidden_state = output.last_hidden_state  # shape: (batch_size, seq_length, bert_hidden_dim)
        # or mean/max pooling
        CLS_token_state = last_hidden_state[:, 0, :]  # obtaining CLS token state which is the first token.
        return CLS_token_state

    def forward(self, batch):
        CLS_hidden_state = self.get_bert_features(batch)
        output = self.arc_margin(CLS_hidden_state, batch['labels'])
        return output
