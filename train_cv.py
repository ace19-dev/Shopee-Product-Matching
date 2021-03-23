'''
https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205
https://modulabs-biomedical.github.io/Bias_vs_Variance

-> low bias, high variance

1st place solution: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/221957
Our final submission first averaged the probabilities of the predicted classes of ViT and ResNext.
This averaged probability vector was then merged with the predicted probabilities of EfficientnetB4 and CropNet
in the second stage. For this purpose, the values were simply summed up.
-> 확율 누적.

'''

import pprint
import random
import timeit

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

import transformer
import model as M
from datasets.product import ProductDataset, NUM_CLASS
from datasets.sampler import ImbalancedDatasetSampler
from option import Options
from training.lr_scheduler import LR_Scheduler
from training.optimizer import Lookahead
from training._loss import FocalLoss2
from training.losses import *
from training.bi_tempered_loss import BiTemperedLogisticLoss
from training.taylor_cross_entropy_loss import TaylorCrossEntropyLoss
from training.metrics import *
from utils.training_helper import *
from utils.image_helper import *

# global variable
best_pred = 0.0
acc_lst_train = []
acc_lst_val = []
lr = 0.0

IMAGE_SIZE = str(transformer.CROP_HEIGHT) + 'x' + \
             str(transformer.CROP_WIDTH)


def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
        [m for m in [8, 14] if m - 1 <= epoch])


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def freeze_until(net, param_name):
    # print([k for k, v in net.named_parameters()])
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


def freeze_bn(net):
    for name, params in net.named_parameters():
        if 'bn' in name:
            params.requires_grad = False
        else:
            params.requires_grad = True


def main():
    global best_pred, acc_lst_train, acc_lst_val, lr

    args, args_desc = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    logger, log_file, final_output_dir, tb_log_dir, create_at = \
        create_logger(args, args_desc, IMAGE_SIZE)
    logger.info('-------------- params --------------')
    logger.info(pprint.pformat(args.__dict__) + '\n')

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    npz_files = os.listdir(os.path.join(args.dataset_root, 'fold'))
    npz_files.sort()
    num = int(len(npz_files) / 2)
    train_npzs = npz_files[:num]
    val_npzs = npz_files[num:]

    def train(epoch):
        global best_pred, acc_lst_train, acc_lst_val

        local_step = 0
        MIXUP_FLAG = False

        losses = AverageMeter()
        accs = AverageMeter()

        model.train()

        # last_time = time.time()
        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (images, targets, fnames) in enumerate(tbar):
            scheduler(optimizer, batch_idx, epoch, best_pred)

            if args.cuda:
                images, targets = images.cuda(), targets.cuda()

            local_step += 1

            # TODO: move cutmix/fmix/mixup func to ProductDataset .py
            # https://www.kaggle.com/sebastiangnther/cassava-leaf-disease-vit-tpu-training
            # CutMix (from https://github.com/clovaai/CutMix-PyTorch)
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                # print('in cutmix')
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

                outputs = model(images)
                # _, outputs = model(images)  # old version
                # feature = model(images)
                # outputs = metric_fc(feature, targets)

                # loss = criterion(activations=outputs,
                #                  labels=torch.nn.functional.one_hot(target_a),
                #                  t1=0.5, t2=1.5) * lam + \
                #        criterion(activations=outputs,
                #                  labels=torch.nn.functional.one_hot(target_b),
                #                  t1=0.5, t2=1.5) * (1. - lam)
                loss = criterion(outputs, target_a) * lam + \
                       criterion(outputs, target_b) * (1. - lam)
            else:
                r = np.random.rand(1)
                if r < args.mixup_prob:
                    MIXUP_FLAG = True
                    # Mixup (from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py)
                    inputs, targets_a, targets_b, lam = mixup_data(images, targets, args.alpha)
                    inputs, targets_a, targets_b = map(Variable, (images, targets_a, targets_b))
                    outputs = model(images)
                    # _, outputs = model(images)  # old version
                    # feature = model(images)
                    # outputs = metric_fc(feature, targets)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    # train_loss += loss.data[0]
                    # _, preds = torch.max(outputs.data, 1)
                    # total += targets.size(0)
                    # correct += (lam * preds.eq(targets_a.data).cpu().sum().float()
                    #             + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float())
                else:
                    MIXUP_FLAG = False
                    outputs = model(images)
                    # _, outputs = model(images)  # old version
                    # feature = model(images)
                    # outputs = metric_fc(feature, targets)
                    # print('outputs:', outputs.shape)
                    # print('targets:', targets.shape)

                    # loss = criterion(activations=outputs,
                    #                  labels=torch.nn.functional.one_hot(targets),
                    #                  t1=0.5, t2=1.5)
                    loss = criterion(outputs, targets)

            _, preds = torch.max(outputs.data, 1)
            # print('preds:', preds)
            # print('targets.data:', targets.data)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # when scheduler lib.
            # scheduler.step()

            batch_size = float(images.size(0))
            losses.update(loss.data, batch_size)
            if MIXUP_FLAG:
                running_corrects = (lam * preds.eq(targets_a.data).cpu().sum().float() +
                                    (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()).long()
                accs.update(running_corrects, batch_size)

            else:
                # https://discuss.pytorch.org/t/trying-to-pass-too-many-cpu-scalars-to-cuda-kernel/87757/4
                accs.update(torch.sum(preds == targets.data), batch_size)

            if batch_idx % 50 == 0:
                tbar.set_description('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), losses.avg, accs.avg))

        logger.info('[Train] Loss: {:.4f} Acc: {:.4f}'.format(losses.avg, accs.avg))

        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', losses.avg, global_steps)
        writer.add_scalar('train_acc', accs.avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    def validate(epoch):
        global best_pred, acc_lst_train, acc_lst_val
        is_best = False

        losses = AverageMeter()
        accs = AverageMeter()

        model.eval()

        tbar = tqdm(val_loader, desc='\r')
        for batch_idx, (images, targets, fnames) in enumerate(tbar):
            if args.cuda:
                images, targets = images.cuda(), targets.cuda()

            with torch.no_grad():
                outputs = model(images)
                # _, outputs = model(images)  # old version
                # feature = model(images)
                # outputs = metric_fc(feature, targets)

                # test_loss += criterion(activations=outputs,
                #                        labels=torch.nn.functional.one_hot(targets),
                #                        t1=0.5, t2=1.5)
                # # ------ TTA ---------
                # images = torch.stack([images, images.flip(-1), images.flip(-2), images.flip(-1, -2),
                #                  images.transpose(-1, -2), images.transpose(-1, -2).flip(-1),
                #                  images.transpose(-1, -2).flip(-2), images.transpose(-1, -2).flip(-1, -2)], 0)
                # images = images.view(-1, 3, transformer.CROP_HEIGHT, transformer.CROP_WIDTH)
                # outputs = model(images)
                # outputs = outputs.view(args.batch_size, 8, -1).mean(1)

                loss = criterion(outputs, targets)
                # get the index of the max log-probability
                pred = outputs.data.max(1, keepdim=True)[1]
                correct = pred.eq(targets.data.view_as(pred)).long().cpu().sum()

                batch_size = float(images.size(0))
                losses.update(loss.data, batch_size)
                accs.update(correct, batch_size)

                # total += images.size(0)
                tbar.set_description('\r[Validate] Loss: %.5f | Top1: %.5f' % (losses.avg, accs.avg))

        logger.info('[Validate] Loss: %.5f | Acc: %.5f' % (losses.avg, accs.avg))

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_acc', accs.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        # save checkpoint
        acc_lst_val += [accs.avg]
        if accs.avg > best_pred:
            logger.info('** [best_pred]: {:.4f}'.format(accs.avg))
            best_pred = accs.avg
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            # 'metric_state_dict': metric_fc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acc_lst_train': acc_lst_train,
            'acc_lst_val': acc_lst_val,
        }, logger=logger, args=args, loss=losses.avg, is_best=is_best,
            image_size=IMAGE_SIZE, create_at=create_at, filename=args.checkpoint_name,
            foldname=valset.fold_name())

    for train_filename, val_filename in zip(train_npzs, val_npzs):
        logger.info('****************************')
        logger.info('%s, training start.' % (train_filename.split('_')[1]))
        logger.info('****************************\n')

        best_pred = 0.0

        args.seed = random.randrange(999)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        logger.info('-------------- seed --------------')
        logger.info(str(args.seed) + '\n')

        logger.info('-------------- image size --------------')
        logger.info(IMAGE_SIZE + '\n')

        trainset = ProductDataset(data_dir=args.dataset_root,
                                  fold=[train_filename],
                                  csv=['train.csv'],
                                  mode='train',
                                  transform=transformer.training_augmentation3(), )
        valset = ProductDataset(data_dir=args.dataset_root,
                                fold=[val_filename],
                                csv=['train.csv'],
                                mode='val',
                                transform=transformer.validation_augmentation())

        logger.info('-------------- train transform --------------')
        logger.info(transformer.training_augmentation3())
        logger.info('\n-------------- valid transform --------------')
        logger.info(transformer.validation_augmentation())

        logger.info('\n-------------- dataset --------------')
        logger.info(trainset)
        logger.info(valset)
        # logger.info('-------------- train transforms --------------')
        # logger.info(pprint.pformat(transforms.training_augmentation().transforms.transforms) + '\n')
        # logger.info('-------------- validation transforms --------------')
        # logger.info(pprint.pformat(transforms.validation_augmentation().transforms.transforms) + '\n')

        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   sampler=ImbalancedDatasetSampler(trainset),
                                                   pin_memory=True,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)

        model = M.Model(backbone=args.model)
        # model.half()  # to save space.
        logger.info('\n-------------- model details --------------')
        logger.info(model)
        # print(model)

        # _in = 1280  # tf_efficientnet_b1_ns
        # _in = 1408  # tf_efficientnet_b2_ns
        # https://github.com/ronghuaiyang/arcface-pytorch/issues/10
        # metric_fc = ArcMarginProduct(_in, NUM_CLASS, s=30, m=0.5, easy_margin=False)

        # freeze_until(model, "pretrained.blocks.5.0.conv_pw.weight")
        # keys = [k for k, v in model.named_parameters() if v.requires_grad]
        # print(keys)
        # freeze_bn(model)

        # criterion and optimizer
        # https://github.com/google/bi-tempered-loss/blob/master/tensorflow/loss_test.py
        # https://github.com/fhopfmueller/bi-tempered-loss-pytorch
        # criterion = BiTemperedLogisticLoss(t1=0.8, t2=1.4, smoothing=0.06)
        # https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
        # criterion = TaylorCrossEntropyLoss(n=6, ignore_index=255, reduction='mean',
        #                                    num_cls=NUM_CLASS, smoothing=0.1)
        # criterion = torch.nn.CrossEntropyLoss()
        # criterion = LabelSmoothingLoss(NUM_CLASS, smoothing=0.1)
        criterion = FocalLoss2()
        # https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
        # criterion = FocalCosineLoss()
        # https://github.com/shengliu66/ELR
        # criterion = elr_loss(num_examp=len(train_loader.dataset), num_classes=NUM_CLASS, _lambda=1, beta=0.9)
        # criterion = elr_plus_loss(num_examp=len(train_loader.dataset), num_classes=5, _lambda=1, beta=0.9)
        logger.info(criterion.__str__())

        # optimizer = torch.optim.SGD(model.parameters(),
        #                             # [{'params': model.parameters()},
        #                             #  {'params': metric_fc.parameters()}],
        #                             # lr=args.lr / _in * args.batch_size,
        #                             lr=args.lr,
        #                             momentum=args.momentum, weight_decay=args.weight_decay)
        # https://github.com/clovaai/AdamP
        from adamp import AdamP
        # optimizer = AdamP([{'params': model.parameters()},
        #                    {'params': metric_fc.parameters()}],
        #                   lr=args.lr, weight_decay=args.weight_decay)
        optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam([{'params': model.parameters()},
        #                   {'params': metric_fc.parameters()}],
        #                  lr=args.lr, weight_decay=args.weight_decay)
        # optimizer = Lookahead(optimizer)

        scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs,
                                 len(train_loader) // args.batch_size,
                                 args.lr_step, warmup_epochs=5)
        # scheduler = \
        #     torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_step_func)

        if args.cuda:
            model.cuda()
            model = nn.DataParallel(model)
            # metric_fc.cuda()
            # metric_fc = nn.DataParallel(metric_fc)
            criterion.cuda()

        # get the number of model parameters
        logger.info('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))

        # check point
        if args.resume is not None:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch'] + 1
                best_pred = checkpoint['best_pred']
                acc_lst_train = checkpoint['acc_lst_train']
                acc_lst_val = checkpoint['acc_lst_val']
                # lst = ['module.pretrained.fc.weight', 'module.pretrained.fc.bias', 'module.head.1.weight',
                #        'module.head.1.bias', 'module.head2.2.weight', 'module.head2.2.bias']
                # pretrained_dict = checkpoint['state_dict']
                # new_model_dict = model.state_dict()
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in lst}
                # new_model_dict.update(pretrained_dict)
                # model.load_state_dict(new_model_dict, strict=False)

                model.load_state_dict(checkpoint['state_dict'], strict=False)
                # metric_fc.load_state_dict(checkpoint['metric_state_dict'], strict=False)

                # original code
                # model.load_state_dict(checkpoint['state_dict'], strict=False)
                # --------------------------
                # w/ external pre-trained
                # --------------------------
                # model.load_state_dict(checkpoint, strict=False)

                # https://github.com/pytorch/pytorch/issues/2830
                if 'optimizer' in checkpoint:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()
                    # optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                raise RuntimeError("=> no resume checkpoint found at '{}'".format(args.resume))

        start = timeit.default_timer()
        for epoch in range(args.start_epoch, args.epochs + 1):
            logger.info('\n\n[%s] ------- Epoch %d -------' % (time.strftime("%Y/%m/%d %H:%M:%S"), epoch))
            train(epoch)
            validate(epoch)

        end = timeit.default_timer()
        logger.info('trained time:%d' % (int((end - start) / 3600)))
        logger.info('%s, training done.\n' % (train_filename.split('_')[1]))
        # logger.info('-------------- Inference Result --------------\n')

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
