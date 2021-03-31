import os
import random
import pandas as pd
from PIL import Image
import cupy
import gc
from tqdm import tqdm
from transformers import BertTokenizer

import cv2
import numpy as np
import torch
import torch.utils.data as data

from tensorflow.keras.preprocessing.sequence import pad_sequences

dataset_root = '../input/shopee-product-matching'
dataset_name = 'product'
modelname = 'bert'
test_batch_size = 8
workers = 4
no_cuda = False
seed = 8

# resume = '../input/checkpoints/(2021-03-17_211032)product_fold3_380x380_tf_efficientnet_b4_ns_acc(54.97810)_loss(0.26047)_checkpoint30.pth.tar'
MODELS = [
    '/home/ace19/my-repo/Shopee-Product-Matching/experiments/shopee-product-matching/bert/(2021-03-30_11:07:28)product_fold1_text_bert_acc(0.00467)_loss(9.30918)_checkpoint3.pth.tar'
]

test_df = pd.read_csv('/home/ace19/dl_data/shopee-product-matching/test.csv')



# from transformers import AutoTokenizer
import math
from pprint import pprint

from transformers import BertModel, BertConfig

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
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        # # self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-05)
        # self.features = nn.BatchNorm1d(in_channels, eps=1e-05)
        # self.flatten = Flatten()
        # nn.init.constant_(self.features.weight, 1.0)
        # self.features.weight.requires_grad = False
        self.classifier = nn.Linear(in_features=768, out_features=self.nclass, bias=True)

    # pooler_output
    def forward(self, x):
        ##################
        # COSINE-SOFTMAX
        ##################
        # x = x.view(-1, num_flat_features(x))
        # x = self.dropout(x)
        # x = self.fc(x)

        features = x
        # Features in rows, normalize axis 1.
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = F.normalize(self.weights, p=2, dim=0, eps=1e-8)
        logits = self.scale.cuda() * torch.mm(features.cuda(), weights_normed.cuda())  # torch.matmul

        return features, logits


class Model(nn.Module):
    def __init__(self, backbone, nclass=11014):
        super(Model, self).__init__()
        self.backbone = backbone
        self.nclass = nclass

        # TODO: use various model
        # self.pretrained = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased",
        #                                                       num_labels=nclass)
        config = BertConfig.from_pretrained('bert-base-multilingual-uncased')
        self.pretrained = BertModel(config)
        # print(self.pretrained)

        in_channels = 768  # BertModel
        self.cosine_softmax = CosineSoftmaxModule(in_channels, nclass)

        # for arcface
        # self.in_features = self.pretrained.classifier.in_features
        self.margin = ArcModule(in_features=in_channels, out_features=nclass)
        # self.bn1 = nn.BatchNorm2d(self.in_features)
        self.dropout = nn.Dropout2d(0.2, inplace=True)
        # self.dropout = nn.Dropout(p=0.4, inplace=True)
        # self.fc1 = nn.Linear(self.in_features * 12 * 12, in_channels)
        self.bn2 = nn.BatchNorm1d(in_channels)

    # cosine-softmax
    def forward(self, input_ids, input_mask):
        if self.backbone.startswith('Bert'):
            outputs = self.pretrained(input_ids=input_ids,
                                      attention_mask=input_mask, )

        return self.cosine_softmax(outputs['pooler_output'])

    # # ArcFace
    # # https://www.kaggle.com/underwearfitting/pytorch-densenet-arcface-validation-training
    # def forward(self, input_ids, input_mask, labels):
    #     if self.backbone.startswith('Bert'):
    #         outputs = self.pretrained(input_ids=input_ids,
    #                                   attention_mask=input_mask, )
    #
    #     # features = self.bn1(x)
    #     # features = self.dropout(features)
    #     # features = features.view(features.size(0), -1)
    #     # features = self.fc1(features)
    #     features = self.bn2(outputs['pooler_output'])
    #     features = F.normalize(features)
    #     if labels is not None:
    #         return self.margin(features, labels)
    #
    #     return features

NUM_CLASS = 11014
MAX_LEN = 512  # TODO: check title max



class ProductTextTestDataset(data.Dataset):
    def __init__(self, data_dir, csv, tokenizer):
        self.data_dir = data_dir
        self.csv = csv
        self.tokenizer = tokenizer

        self.df = pd.concat([pd.read_csv(data_dir + '/%s' % f) for f in self.csv])
        self.posting_id = self.df['posting_id'].values.tolist()

        sentences = self.df['title']
        # BERT 입력 형식에 맞게 변환
        sentences = ["[CLS] " + str(s) + " [SEP]" for s in sentences]
        tokenized_texts = [tokenizer.tokenize(s) for s in sentences]
        # 토큰을 숫자 인덱스로 변환
        self.input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
        self.input_ids = pad_sequences(self.input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

        self.attention_masks = []
        # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
        # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
        for seq in self.input_ids:
            seq_mask = [float(i > 0) for i in seq]
            self.attention_masks.append(seq_mask)

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]), \
               torch.tensor(self.attention_masks[index]), self.posting_id[index]

    def __len__(self):
        return len(self.input_ids)


global best_pred, acc_lst_train, acc_lst_val

cuda = not no_cuda and torch.cuda.is_available()
print(cuda)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

model = Model(backbone='Bert', nclass=NUM_CLASS)
print('\n-------------- model details --------------')
print(model)

if cuda:
    model.cuda()
    model = nn.DataParallel(model)


CHUNK = 1024 * 4
print('Computing image embeddings...')
CTS = len(test_df) // CHUNK

if len(test_df) % CHUNK != 0:
    CTS += 1

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=False)

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
            model.load_state_dict(checkpoint['state_dict'])
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

        testset = ProductTextTestDataset(data_dir=dataset_root,
                                         csv=test_df.iloc[a:b],
                                         tokenizer=tokenizer)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=workers)

        features_list = []
        tbar = tqdm(test_loader, desc='\r')
        for batch_idx, (input_ids, input_mask, pos_id) in enumerate(tbar):
            if cuda:
                input_ids, input_mask, labels = input_ids.cuda(), input_mask.cuda()

            with torch.no_grad():
                # # ArcFace
                # outputs = model(input_ids, input_mask, labels)
                # cosine-softmax
                features, _ = model(input_ids, input_mask)

                features_list.append(features.cpu().numpy())

        embeds.extend(features_list)

    features_pool.append(np.concatenate(embeds))
    _ = gc.collect()

# -------------
# max pooling.
# -------------
text_features = features_pool[0]
for i in range(1, len(features_pool)):
    text_features = np.maximum(text_features, features_pool[i])

print('text embeddings shape', text_features.shape)


preds = []
CHUNK = 1024

print('Finding similar titles...')
CTS = len(test) // CHUNK
if len(test) % CHUNK != 0:
    CTS += 1

for j in range(CTS):

    a = j * CHUNK
    b = (j + 1) * CHUNK
    b = min(b, len(test))
    print('chunk', a, 'to', b)

    cts = cupy.matmul(text_features, text_features[a:b].T).T

    for k in range(b - a):
        IDX = cupy.where(cts[k,] > 0.7)[0]
        o = test.iloc[cupy.asnumpy(IDX)].posting_id.values
        preds.append(o.tolist())

del text_features
_ = gc.collect()


submit_df = pd.read_csv("/home/ace19/dl_data/shopee-product-matching/sample_submission.csv")

submit_df['matches'] = preds
submit_df[['posting_id', 'matches']].to_csv('submission.csv', index=False)

sub = pd.read_csv('submission.csv')
sub.head()

