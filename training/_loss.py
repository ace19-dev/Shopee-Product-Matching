'''
From https://github.com/PistonY/torch-toolbox
'''

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """This is label smoothing loss function.
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = smooth_one_hot(target, self.cls, self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


@torch.no_grad()
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    Warning: This function has no grad.
    """
    # assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))

    smooth_label = torch.empty(size=label_shape, device=true_labels.device)
    smooth_label.fill_(smoothing / (classes - 1))
    smooth_label.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return smooth_label


class FocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.weight)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class FocalLoss2(nn.Module):
    def __init__(self, weight=None, alpha=1, gamma=2, reduce=True):
        super(FocalLoss2, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        loss = F.cross_entropy(inputs, targets, weight=self.weight, )

        pt = torch.exp(-loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


# https://github.com/shengliu66/ELR
class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, _lambda=3, beta=0.7):
        r"""Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """

        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp,
                                                                                                        self.num_classes)
        self.beta = beta
        self._lambda = _lambda

    def forward(self, index, output, label):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, used to track training examples in different iterations.
         * `output` Model's prediction, same as PyTorch provided functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """

        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
                (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self._lambda * elr_reg
        return final_loss


# https://github.com/shengliu66/ELR
class elr_plus_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, _lambda=1, beta=0.9):
        super(elr_plus_loss, self).__init__()
        # self.config = config
        self._lambda = _lambda
        # self.pred_hist = (torch.zeros(num_examp, num_classes)).cuda()
        self.q = 0
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, output, y_labeled):
        y_pred = F.softmax(output, dim=1)

        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled * self.q
            y_labeled = y_labeled / (y_labeled).sum(dim=1, keepdim=True)

        ce_loss = torch.mean(-torch.sum(torch.unsqueeze(y_labeled, dim=1) * F.log_softmax(output, dim=1), dim=-1))
        reg = ((1 - (self.q * y_pred).sum(dim=1)).log()).mean()
        # final_loss = ce_loss + sigmoid_rampup(iteration, 0) * (self._lambda * reg)
        final_loss = ce_loss + 1.0 * (self._lambda * reg)

        # return final_loss, y_pred.cpu().detach()
        return final_loss

    def update_hist(self, epoch, out, index=None, mix_index=..., mixup_l=1):
        y_pred_ = F.softmax(out, dim=1)
        self.pred_hist[index] = self.beta * self.pred_hist[index] + (1 - self.beta) * y_pred_ / (y_pred_).sum(dim=1,
                                                                                                              keepdim=True)
        self.q = mixup_l * self.pred_hist[index] + (1 - mixup_l) * self.pred_hist[index][mix_index]


# https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
class CosineLoss(nn.Module):
    def __init__(self, xent=.1, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y,
                                              reduction=self.reduction)
        cent_loss = F.cross_entropy(F.normalize(input), target, reduction=self.reduction)

        return cosine_loss + self.xent * cent_loss


# https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

        self.xent = xent

        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target, reduction="mean"):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y,
                                              reduction=reduction)

        cent_loss = F.cross_entropy(F.normalize(input), target, reduce=False)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        return cosine_loss + self.xent * focal_loss


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def sigmoid_rampdown(current, rampdown_length):
    """Exponential rampdown"""
    if rampdown_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampdown_length)
        phase = 1.0 - (rampdown_length - current) / rampdown_length
        return float(np.exp(-12.5 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def linear_rampdown(current, rampdown_length):
    """Linear rampup"""
    assert current >= 0 and rampdown_length >= 0
    if current >= rampdown_length:
        return 1.0
    else:
        return 1.0 - current / rampdown_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    current = np.clip(current, 0.0, rampdown_length)
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def cosine_rampup(current, rampup_length):
    """Cosine rampup"""
    current = np.clip(current, 0.0, rampup_length)
    return float(-.5 * (np.cos(np.pi * current / rampup_length) - 1))


class SymmetricCrossEntropy(nn.Module):

    def __init__(self, alpha=0.1, beta=1.0, num_classes=5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, logits, targets, reduction='mean'):
        onehot_targets = torch.eye(self.num_classes)[targets].cuda()
        ce_loss = F.cross_entropy(logits, targets, reduction=reduction)
        rce_loss = (-onehot_targets * logits.softmax(1).clamp(1e-7, 1.0).log()).sum(1)
        if reduction == 'mean':
            rce_loss = rce_loss.mean()
        elif reduction == 'sum':
            rce_loss = rce_loss.sum()
        return self.alpha * ce_loss + self.beta * rce_loss


