from __future__ import print_function

import csv
import inspect
import os
import random

import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm

# import models.model_zoo as models
import transformer
import model as M
from option import Options
from datasets.product import NUM_CLASS, ProductTestDataset, ProductDataset
from training._loss import *
from training.taylor_cross_entropy_loss import TaylorCrossEntropyLoss
from utils.training_helper import AverageMeter

# global variable
best_pred = 0.0
acclist_train = []
acclist_val = []

__cwd__ = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

MODELS = [
    'experiments/shopee-product-matching/seresnext50_32x4d/(2021-01-17_13:48:23)shopee-product-matching_576x576_seresnext50_32x4d_acc(88.53954)_loss(0.02971)_checkpoint14.pth.tar',
    'experiments/shopee-product-matching/seresnext50_32x4d/(2021-01-17_16:17:56)shopee-product-matching_576x576_seresnext50_32x4d_acc(88.21989)_loss(0.02985)_checkpoint16.pth.tar',
    'experiments/shopee-product-matching/seresnext50_32x4d/(2021-01-17_18:30:10)shopee-product-matching_576x576_seresnext50_32x4d_acc(88.25268)_loss(0.02202)_checkpoint20.pth.tar',
    'experiments/shopee-product-matching/seresnext50_32x4d/(2021-01-17_21:21:19)shopee-product-matching_576x576_seresnext50_32x4d_acc(88.37082)_loss(0.02182)_checkpoint11.pth.tar',
    'experiments/shopee-product-matching/seresnext50_32x4d/(2021-01-17_23:58:02)shopee-product-matching_576x576_seresnext50_32x4d_acc(88.06683)_loss(0.02237)_checkpoint20.pth.tar',
]


def main():
    global best_pred, acclist_train, acclist_val

    args, _ = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    testset = ProductTestDataset(args.dataset_root,
                                 transform=transformer.test_augmentation())
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    # init the model
    model = M.Model(NUM_CLASS, backbone=args.model)
    print(model)

    # https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
    loss = TaylorCrossEntropyLoss(n=4, ignore_index=255, reduction='mean',
                                  num_cls=NUM_CLASS, smoothing=0.1)
    # # loss = FocalLoss2()
    # # https://github.com/shengliu66/ELR
    # # loss = elr_loss(num_examp=len(test_loader.dataset), num_classes=NUM_CLASS, _lambda=1, beta=0.9)
    # # loss = nn.CrossEntropyLoss()
    # # # loss = LabelSmoothingLoss(args.nclass, smoothing=0.1)

    if args.cuda:
        model.cuda()
        loss.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = nn.DataParallel(model)

    test_result = []
    for idx, pretrained in enumerate(MODELS):
        if pretrained is not None:
            if os.path.isfile(pretrained):
                print("=> loading checkpoint '{}'".format(pretrained))
                checkpoint = torch.load(pretrained)
                args.start_epoch = checkpoint['epoch'] + 1
                best_pred = checkpoint['best_pred']
                acclist_train = checkpoint['acclist_train']
                acclist_val = checkpoint['acclist_val']
                model.module.load_state_dict(checkpoint['state_dict'], strict=False)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(pretrained, checkpoint['epoch']))
            else:
                raise RuntimeError("=> no resume checkpoint found at '{}'". \
                                   format(pretrained))
        else:
            raise RuntimeError("=> config \'MODELS[i]\' is '{}'". \
                               format(pretrained))

        test_infos, test_probs = test(idx, args, test_loader, model, loss, pretrained)
        test_result.append(test_probs)

    test_result = np.array(test_result)
    # print('test_result >> \n', test_result)
    test_result = np.transpose(test_result, (1, 0, 2))
    # print('\nensemble >> \n', ensemble)
    probs_ensemble = np.mean(test_result, axis=1)
    # print('\nmean >> \n', ensemble)
    pred = np.argmax(probs_ensemble, axis=1)
    pred = np.expand_dims(pred, axis=1)
    # print('\nargmax >> \n', pred)

    ensemble = np.concatenate((np.array(test_infos), pred, probs_ensemble), axis=1)
    # print('\ensemble >> \n', ensemble)
    ensemble = ensemble.tolist()

    # confusion_matrix = torch.zeros(args.nclass, args.nclass)
    # for idx, item in enumerate(ensemble):
    #     confusion_matrix[torch.tensor([int(item[1])], dtype=torch.int64),
    #                      torch.tensor([int(item[2])], dtype=torch.int64)] += 1
    #
    # print('\nEnsemble result')
    # print('----------------------------------\n')
    # print('confusion matrix:\n', confusion_matrix.numpy().astype(int))
    # # get the per-class accuracy
    # print('\nper-class accuracy:\n', confusion_matrix.diag() / confusion_matrix.sum(1))
    # print('\n----------------------------------\n')

    # create result file.
    if args.result is not None:
        if not os.path.exists(args.result):
            os.makedirs(args.result)

        fout = open(args.result + '/%s_result.csv' % time.strftime("%Y-%m-%d_%H:%M:%S"), 'w',
                    encoding='UTF-8', newline='')
        writer = csv.writer(fout)
        writer.writerow(['filename', 'gt', 'predict', 'heaven_prob', 'earth_prob', 'good_prob', 'raw_prob'])
        for r in sorted(ensemble):
            writer.writerow([r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
        fout.close()

    # print('\n----------------------------------\n')
    # print('confusion matrix:\n', confusion_matrix.numpy().astype(int))
    # # get the per-class accuracy
    # print('\nper-class accuracy:\n', confusion_matrix.diag() / confusion_matrix.sum(1))
    # print('\n----------------------------------\n')


def test(model_idx, args, test_loader, model, loss, pretrained):
    global best_pred, acclist_train, acclist_val
    infos_result = []
    probabilities_result = []

    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    confusion_matrix = torch.zeros(NUM_CLASS, NUM_CLASS)

    model.eval()

    tbar = tqdm(test_loader, desc='\r')
    for batch_idx, (images, targets, fnames, _) in enumerate(tbar):
        # images = torch.stack(images, dim=1)
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()

        with torch.no_grad():
            output = model(images)
            # print(output)
            # print(targets)
            # l = loss(output, targets)
            # losses.update(l.item(), images.size(0))
            # acc1, acc5 = accuracy(output, targets, topk=(1, 1))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            _, preds = torch.max(output.data, 1)
            # probs = torch.softmax(output, dim=1)
            print('')

            # for idx, (t, p) in enumerate(zip(targets.view(-1), preds.view(-1))):
            #     confusion_matrix[t.long(), p.long()] += 1
            #
            #     infos = []
            #     infos.append(os.path.split(fnames[0][idx])[0])
            #     infos.append(t.long().cpu().numpy())
            #     # infos.append(p.long().cpu().numpy())
            #     infos_result.append(infos)
            #
            #     probabilities = []
            #     probabilities.append(probs[idx][0].cpu().numpy())
            #     probabilities.append(probs[idx][1].cpu().numpy())
            #     probabilities.append(probs[idx][2].cpu().numpy())
            #     probabilities.append(probs[idx][3].cpu().numpy())
            #     probabilities_result.append(probabilities)

        tbar.set_description('\r[Test] Loss: %.3f | Top1: %.3f' % (losses.avg, top1.avg))

    torch.cuda.empty_cache()

    print('\n----------------------------------')
    print('{}/{}th pretrained model: {}'.format(model_idx, len(MODELS), pretrained))
    # print('confusion matrix:\n', confusion_matrix.numpy().astype(int))
    # # get the per-class accuracy
    # print('\nper-class accuracy:\n', confusion_matrix.diag() / confusion_matrix.sum(1))
    print('----------------------------------\n')

    return infos_result, probabilities_result


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
