import timm
from pprint import pprint

from training.metric_learning_losses import *


class ShopeeNet(nn.Module):
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
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ShopeeNet, self).__init__()

        self.model_name = model_name

        print('Building Model Backbone for {} model'.format(model_name))
        model_names = timm.list_models(pretrained=True)
        pprint(model_names)

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        pprint(self.backbone)

        if model_name.startswith('seresnext50'):
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            # self.backbone.global_pool = nn.Identity()

        elif model_name.startswith('tf_efficientnet_b'):
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            # self.backbone.global_pool = nn.Identity()

        elif model_name.startswith('dm_nfnet_f'):
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            # self.backbone.head.global_pool = nn.Identity()

        elif model_name.startswith('resnest'):
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            # self.backbone.global_pool = nn.Identity()

        # self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        elif loss_module == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin)
        elif loss_module == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.loss_module in ('arcface', 'cosface', 'adacos'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return feature, logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        # x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x

    def __str__(self):
        return 'model_name: {}, use_fc: {}'.format(self.model_name, self.use_fc)
