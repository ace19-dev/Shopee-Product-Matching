from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import sys
import time
from sklearn.metrics import roc_auc_score, average_precision_score

np.set_printoptions(threshold=sys.maxsize)

import torch
import torch.nn as nn
import torch.distributed as dist

float_formatter = "{:.8f}".format
np.set_printoptions(formatter={'float_kind': float_formatter},
                    threshold=sys.maxsize)


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def create_logger(args, args_desc, image_size=None):
    root_output_dir = Path(args.output)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    # dataset = args.dataset
    model = args.model
    cfg_name = os.path.basename(args_desc)

    final_output_dir = root_output_dir / cfg_name / model

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d_%H:%M:%S')
    if image_size is not None:
        log_file = '({}){}_{}_{}_lr({}).log'.format(time_str, cfg_name, image_size, args.model, args.lr)
    else:
        log_file = '({}){}_{}_lr({}).log'.format(time_str, cfg_name, args.model, args.lr)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(args.tb_log) / cfg_name / model / log_file
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, log_file, str(final_output_dir), str(tensorboard_log_dir), time_str


# def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
#     lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
#     optimizer.param_groups[0]['lr'] = lr
#     return lr


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         prob, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res


def display_data(fname, image):
    # display image to verify
    image = image.numpy()
    image = np.transpose(image, (0, 2, 3, 1))
    # # assert not np.any(np.isnan(image))
    n_batch = image.shape[0]
    # n_view = train_batch_xs.shape[1]
    for i in range(n_batch):
        img = image[i]
        # scipy.misc.toimage(img).show() Or
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite('/home/ace19/Pictures/' + fname[i].split('/')[-1], img)
        # cv2.imshow(str(train_batch_ys[idx]), img)
        cv2.waitKey(100)
        cv2.destroyAllWindows()


def save_checkpoint(state, logger, args, loss, is_best, create_at, filename, foldname, image_size=None):
    _filename = '(' + create_at + ')' + filename + '_%s_%s_%s_acc(%s)_loss(%s)_checkpoint%s.pth.tar'
    # _filename = filename + '_checkpoint.pth.tar'

    """Saves checkpoint to disk"""
    directory = "%s/%s/%s/" % (args.output, 'shopee-product-matching', args.model)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # save file per epoch
    if image_size is not None:
        _filename = directory + _filename % (foldname, image_size, args.model,
                                             str(format(state['acc_lst_val'][-1], ".5f")),
                                             round(loss, 5), str(state['epoch']))
    else:
        _filename = directory + _filename % (foldname, 'text', args.model,
                                             str(format(state['acc_lst_val'][-1], ".5f")),
                                             round(loss, 5), str(state['epoch']))
    # save file
    torch.save(state, _filename)
    logger.info('Saving checkpoint to {}'.format(_filename))
    if is_best:
        shutil.copyfile(_filename, directory + filename + '_model_best.pth.tar')


def display_data(data, fnames):
    # display image to verify
    data = data.numpy()
    data = np.transpose(data, (0, 2, 3, 1))
    # # assert not np.any(np.isnan(data))
    n_batch = data.shape[0]
    # n_view = train_batch_xs.shape[1]
    for i in range(n_batch):
        img = data[i]
        # scipy.misc.toimage(img).show() Or
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite('sample/' + fnames[i].split('/')[-1], img)
        # cv2.imshow(str(train_batch_ys[idx]), img)
        # cv2.waitKey(100)
        # cv2.destroyAllWindows()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        prob, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
