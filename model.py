import torch.nn as nn

import torch
from torch.nn import functional as F
import torchvision.models as torch_models

import timm
from pprint import pprint


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_feature = 1
    for s in size:
        num_feature *= s

    return num_feature


class Model(nn.Module):
    def __init__(self, backbone, nclass=11014):
        super(Model, self).__init__()
        self.backbone = backbone
        self.ncode = 32
        self.nclass = nclass

        model_names = timm.list_models(pretrained=True)
        pprint(model_names)
        self.pretrained = timm.create_model(self.backbone, pretrained=True, num_classes=nclass)
        # # Below code is used when if pretrained is False
        # pre_model = torch.load('/home/ace19/.cache/torch/checkpoints/resnest101-22405ba7.pth')
        # del pre_model['fc.weight']
        # del pre_model['fc.bias']
        # self.pretrained.load_state_dict(pre_model, strict=False)

        # # copying modules from pretrained models
        # if self.backbone == 'resnet101':
        #     self.pretrained = torch_models.resnet101(pretrained=True)
        # elif self.backbone == 'resnet50':
        #     self.pretrained = torch_models.resnet50(pretrained=True)
        # elif self.backbone.startswith('tf_efficientnet'):
        #     # old efficientnet
        #     # self.pretrained = EfficientNet.from_pretrained(self.backbone, num_classes=nclass)
        #
        #     # use noisy-student pretrained
        #     model_names = timm.list_models(pretrained=True)
        #     pprint(model_names)
        #     self.pretrained = timm.create_model(self.backbone, pretrained=True, num_classes=nclass)
        #     # print('')
        # else:
        #     raise RuntimeError('unknown backbone: {}'.format(self.backbone))

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

        self.weights = torch.nn.Parameter(torch.randn(in_channels, self.nclass))
        self.scale = torch.nn.Parameter(F.softplus(torch.randn(())))
        self.fc = nn.Linear(in_channels, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    # def forward(self, x):
    #     if self.backbone.startswith('tf_efficientnet'):
    #         # x = self.pretrained.extract_features(x)
    #         x = self.pretrained.conv_stem(x)
    #         x = self.pretrained.bn1(x)
    #         x = self.pretrained.act1(x)
    #         x = self.pretrained.blocks(x)
    #         x = self.pretrained.conv_head(x)
    #         x = self.pretrained.bn2(x)
    #         x = self.pretrained.act2(x)
    #         # print(x.shape)
    #         x = self.pretrained.global_pool(x)
    #         return self.pretrained.classifier(x)
    #
    #     elif self.backbone.startswith('resnet') or \
    #             self.backbone.startswith('resnext') or \
    #             self.backbone.startswith('seresnext') or \
    #             self.backbone.startswith('resnest'):
    #         x = self.pretrained.conv1(x)
    #         x = self.pretrained.bn1(x)
    #         x = self.pretrained.act1(x)
    #         x = self.pretrained.maxpool(x)
    #         x = self.pretrained.layer1(x)
    #         x = self.pretrained.layer2(x)
    #         x = self.pretrained.layer3(x)
    #         x = self.pretrained.layer4(x)
    #         # x = self.pretrained.global_pool(x)
    #
    #     return self.head2(x)
    #     # return self.head(x)

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
        x = F.dropout2d(x, p=0.2)
        x = self.fc(x)

        features = x
        # Features in rows, normalize axis 1.
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        # Mean vectors in colums, normalize axis 0.
        weights_normed = F.normalize(self.weights, p=2, dim=0, eps=1e-8)
        logits = self.scale.cuda() * torch.mm(features.cuda(), weights_normed.cuda())  # torch.matmul

        return features, logits
