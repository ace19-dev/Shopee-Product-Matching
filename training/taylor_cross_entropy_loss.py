'''
From https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

'''
proposed in this paper: [Exploring Alternatives to Softmax Function](https://arxiv.org/pdf/2011.11538.pdf)
'''


##
# version 1: use torch.autograd
class TaylorSoftmax(nn.Module):
    '''
    This is the autograd version
    '''

    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        '''
        usage similar to nn.Softmax:
            >>> mod = TaylorSoftmax(dim=1, n=4)
            >>> inten = torch.randn(1, 32, 64, 64)
            >>> out = mod(inten)
        '''
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)

        return out


##
# version 1: use torch.autograd
class TaylorCrossEntropyLoss(nn.Module):
    '''
    This is the autograd version
    '''

    def __init__(self, n=2, ignore_index=-1, reduction='mean', num_cls=5, smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.lab_smooth = LabelSmoothingLoss(num_cls, smoothing=smoothing)

    def forward(self, logits, labels):
        '''
        usage similar to nn.CrossEntropyLoss:
            >>> crit = TaylorCrossEntropyLoss(n=4)
            >>> inten = torch.randn(1, 10, 64, 64)
            >>> label = torch.randint(0, 10, (1, 64, 64))
            >>> out = crit(inten, label)
        '''
        log_probs = self.taylor_softmax(logits).log()
        # TaylorCE loss
        # loss = F.nll_loss(log_probs, labels, reduction=self.reduction,
        #                   ignore_index=self.ignore_index)
        # https://www.kaggle.com/yerramvarun/cassava-taylorce-loss-label-smoothing-combo
        # Combine TaylorCE loss + Label Smoothing Combo
        loss = self.lab_smooth(log_probs, labels)

        return loss

    def __str__(self):
        print('smoothing: ' + str(self.smoothing))
        print('self.taylor_softmax: ' + str(self.taylor_softmax))


class LabelSmoothingLoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """Taylor Softmax and log are already applied on the logits"""
        # pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# Sample
# if __name__ == '__main__':
#     import torchvision
#
#     tsoftmax = TaylorSoftmax(dim=0, n=4)
#     inten = torch.randn(3, 4, 5, 6)
#     out = tsoftmax(inten)
#     print(out.size())
#     print(out[:, 0, 0, :4])
#     print(out[:, 0, 0, :4].sum(dim=0))
#
#
#     class Model(nn.Module):
#         def __init__(self):
#             super(Model, self).__init__()
#             net = torchvision.models.resnet18(pretrained=False)
#             self.conv1 = net.conv1
#             self.bn1 = net.bn1
#             self.maxpool = net.maxpool
#             self.relu = net.relu
#             self.layer1 = net.layer1
#             self.layer2 = net.layer2
#             self.layer3 = net.layer3
#             self.layer4 = net.layer4
#             self.fc = nn.Conv2d(512, 19, 3, 1, 1)
#
#         def forward(self, x):
#             feat = self.conv1(x)
#             feat = self.bn1(feat)
#             feat = self.relu(feat)
#             feat = self.maxpool(feat)
#             feat = self.layer1(feat)
#             feat = self.layer2(feat)
#             feat = self.layer3(feat)
#             feat = self.layer4(feat)
#             feat = self.fc(feat)
#             out = F.interpolate(feat, x.size()[2:], mode='bilinear', align_corners=True)
#             return out
#
#
#     red = 'mean'
#     bs = 64
#     net1 = Model()
#     net1.cuda()
#     net1.train()
#     criteria1 = TaylorCrossEntropyLoss(n=4, ignore_index=255, reduction=red)
#
#     optim1 = torch.optim.SGD(net1.parameters(), lr=1e-2)
#
#     for it in range(300):
#         inten = torch.randn(bs, 3, 224, 224).cuda()
#         lbs = torch.randint(0, 19, (bs, 224, 224)).cuda()
#         lbs[1, 1, 1] = 255
#         lbs[30, 3, 2:200] = 255
#         lbs[18, 4:7, 8:200] = 255
#         logits = net1(inten)
#
#         loss1 = criteria1(logits, lbs)
#         optim1.zero_grad()
#         loss1.backward()
#         optim1.step()
#         print(loss1.item())
