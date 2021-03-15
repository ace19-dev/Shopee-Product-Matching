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
from datasets.product import ProductDataset
from datasets.sampler import ImbalancedDatasetSampler
from option import Options
from training.lr_scheduler import LR_Scheduler
from training.optimizer import Lookahead
from training.loss import FocalLoss2
from training.bi_tempered_loss import BiTemperedLogisticLoss
from training.taylor_cross_entropy_loss import TaylorCrossEntropyLoss
from utils.training_helper import *
from utils.image_helper import *

# global variable
best_pred = 0.0
acc_lst_train = []
acc_lst_val = []
lr = 0.0

IMAGE_SIZE = str(transformer.CROP_HEIGHT) + 'x' + \
             str(transformer.CROP_WIDTH)


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
    train_npzs = npz_files[:5]
    val_npzs = npz_files[5:]

    def train(epoch):
        global best_pred, acc_lst_train, acc_lst_val

        model.train()

        running_loss = 0.0
        running_corrects = 0
        total_batch_size = 0
        local_step = 0
        MIXUP_FLAG = False

        # last_time = time.time()
        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (images, targets, fnames, indexs) in enumerate(tbar):
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
                # Crop n Stack random 3 sample model
                # images2 = crop_images(images, 384, 4)
                # compute output
                _, outputs = model(images)

                # # ------ TTA - --------
                # batch_size, n_crops, c, h, w = images.size()
                # # fuse batch size and ncrops
                # outputs = model(images.view(-1, c, h, w))
                # # avg over crops
                # outputs = outputs.view(batch_size, n_crops, -1).mean(1)
                # # max over crops
                # # outputs = torch.max(outputs.view(batch_size, n_crops, -1), 1).values
                # # --------------------
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
                    _, outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    # train_loss += loss.data[0]
                    # _, preds = torch.max(outputs.data, 1)
                    # total += targets.size(0)
                    # correct += (lam * preds.eq(targets_a.data).cpu().sum().float()
                    #             + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float())
                else:
                    MIXUP_FLAG = False
                    # [batch_size, n_crops, c, h, w]
                    # Crop n Stack random 3 sample model
                    # images2 = crop_images(images, 384, 4)
                    _, outputs = model(images)
                    # print('outputs:', outputs.shape)
                    # print('targets:', targets.shape)

                    # TTA by - https://www.kaggle.com/japandata509/ensemble-resnext50-32x4d-efficientnet-0-903
                    # def tta_inference_func(test_loader):
                    #     model.eval()
                    #     bar = tqdm(test_loader)
                    #     PREDS = []
                    #     LOGITS = []
                    #
                    #     with torch.no_grad():
                    #         for batch_idx, images in enumerate(bar):
                    #             x = images.to(device)
                    #             x = torch.stack([x, x.flip(-1), x.flip(-2), x.flip(-1, -2),
                    #                              x.transpose(-1, -2), x.transpose(-1, -2).flip(-1),
                    #                              x.transpose(-1, -2).flip(-2), x.transpose(-1, -2).flip(-1, -2)], 0)
                    #             x = x.view(-1, 3, image_size, image_size)
                    #             logits = model(x)
                    #             logits = logits.view(BATCH_SIZE, 8, -1).mean(1)
                    #             PREDS += [torch.softmax(logits, 1).detach().cpu()]
                    #             LOGITS.append(logits.cpu())
                    #
                    #         PREDS = torch.cat(PREDS).cpu().numpy()
                    #
                    #     return PREDS

                    # # ------ TTA by transformer --------
                    # # batch_size, n_crops, c, h, w = images.size()
                    # images2 = five_crop(images, 256)
                    # n_crops, batch_size, c, h, w = images2.size()
                    # # fuse batch size and ncrops
                    # outputs = model(images.view(-1, c, h, w))
                    # # print(outputs.size())
                    # # avg over crops
                    # # outputs = outputs.view(batch_size, n_crops, -1).mean(1)
                    # # print(outputs.size())
                    # # max over crops
                    # outputs = torch.max(outputs.view(batch_size, n_crops, -1), 1).values
                    # # --------------------

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

            batch_size = images.size(0)
            # statistics
            running_loss += loss.item() * batch_size
            if MIXUP_FLAG:
                running_corrects += (lam * preds.eq(targets_a.data).cpu().sum().float()
                                     + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()).long()
            else:
                running_corrects += torch.sum(preds == targets.data)
            total_batch_size += batch_size

            if batch_idx % 50 == 0:
                tbar.set_description('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), running_loss / float(total_batch_size),
                           running_corrects / float(total_batch_size)))

        epoch_loss = running_loss / float(len(train_loader.dataset))
        epoch_acc = running_corrects / float(len(train_loader.dataset))

        logger.info('[Train] Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', epoch_loss, global_steps)
        writer.add_scalar('train_acc', epoch_acc, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    def validate(epoch):
        global best_pred, acc_lst_train, acc_lst_val
        is_best = False
        # confusion_matrix = torch.zeros(NUM_CLASS, NUM_CLASS)

        test_loss = 0
        correct = 0
        total = 0

        model.eval()

        tbar = tqdm(val_loader, desc='\r')
        for batch_idx, (images, targets, fnames, indexs) in enumerate(tbar):
            if args.cuda:
                images, targets = images.cuda(), targets.cuda()

            with torch.no_grad():
                # image shape  [n_crops, batch_size, c, h, w]
                # Crop n Stack random 3 sample model
                # images2 = crop_images(images, 384, 4)
                _, outputs = model(images)

                # # ------ TTA 1 ---------
                # batch_size, n_crops, c, h, w = images.size()
                # # images2 = five_crop(images, 288)
                # # n_crops, batch_size, c, h, w = images2.size()
                # # fuse batch size and ncrops
                # outputs = model(images.view(-1, c, h, w))
                # # avg over crops
                # # outputs = outputs.view(batch_size, n_crops, -1).mean(1)
                # # max over crops
                # outputs = torch.max(outputs.view(batch_size, n_crops, -1), 1).values
                # # --------------------
                # test_loss += criterion(activations=outputs,
                #                        labels=torch.nn.functional.one_hot(targets),
                #                        t1=0.5, t2=1.5)
                # # ------ TTA 2 ---------
                # images = torch.stack([images, images.flip(-1), images.flip(-2), images.flip(-1, -2),
                #                  images.transpose(-1, -2), images.transpose(-1, -2).flip(-1),
                #                  images.transpose(-1, -2).flip(-2), images.transpose(-1, -2).flip(-1, -2)], 0)
                # images = images.view(-1, 3, transformer.CROP_HEIGHT, transformer.CROP_WIDTH)
                # outputs = model(images)
                # outputs = outputs.view(args.batch_size, 8, -1).mean(1)

                test_loss += criterion(outputs, targets).item()
                # test_loss += criterion(indexs.cpu().detach().numpy().tolist(), outputs, targets).item()

                # get the index of the max log-probability
                pred = outputs.data.max(1, keepdim=True)[1]
                correct += pred.eq(targets.data.view_as(pred)).long().cpu().sum()
                total += images.size(0)

                # confusion matrix
                # _, preds = torch.max(outputs, 1)
                # for t, p in zip(targets.view(-1), preds.view(-1)):
                #     confusion_matrix[t.long(), p.long()] += 1
                tbar.set_description('\r[Validate] Loss: %.5f | Top1: %.5f' %
                                     (test_loss / float(total), correct / float(total)))

        test_loss /= float(len(val_loader.dataset))
        test_acc = 100. * correct / float(len(val_loader.dataset))

        logger.info('[Validate] Loss: %.5f | Acc: %.5f' % (test_loss, test_acc))
        # logger.info('\n----------------------------------')
        # logger.info('confusion matrix:\n{}'.format(confusion_matrix.numpy().astype(int)))
        # # per-class accuracy
        # logger.info('\nper-class accuracy(precision):\n{}'.format(confusion_matrix.diag() / confusion_matrix.sum(1)))
        # logger.info('----------------------------------')

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', test_loss, global_steps)
        writer.add_scalar('valid_acc', test_acc, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        # save checkpoint
        acc_lst_val += [test_acc]
        if test_acc > best_pred:
            logger.info('** [best_pred]: {:.4f}'.format(test_acc))
            best_pred = test_acc
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acc_lst_train': acc_lst_train,
            'acc_lst_val': acc_lst_val,
            }, logger=logger, args=args, loss=test_loss, is_best=is_best,
            image_size=IMAGE_SIZE, create_at=create_at, filename=args.checkpoint_name,
            foldname=valset.fold_name())

    for train_filename, val_filename in zip(train_npzs  , val_npzs):
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
                                  transform=transformer.training_augmentation3(),
                                  )
        valset = ProductDataset(data_dir=args.dataset_root,
                                fold=[val_filename],
                                csv=['train.csv'],
                                mode='val',
                                transform=transformer.validation_augmentation()
                                )

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

        # freeze_until(model, "pretrained.blocks.5.0.conv_pw.weight")
        # keys = [k for k, v in model.named_parameters() if v.requires_grad]
        # print(keys)

        # freeze_bn(model)

        # criterion and optimizer
        # https://github.com/google/bi-tempered-loss/blob/master/tensorflow/loss_test.py
        # https://github.com/fhopfmueller/bi-tempered-loss-pytorch
        # criterion = BiTemperedLogisticLoss(t1=0.8, t2=1.4, smoothing=0.06)
        # https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
        criterion = TaylorCrossEntropyLoss(n=6, ignore_index=255, reduction='mean',
                                           num_cls=11014, smoothing=0.1)

        # criterion = LabelSmoothingLoss(NUM_CLASS, smoothing=0.1)
        # criterion = FocalLoss2()
        # https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
        # criterion = FocalCosineLoss()
        # https://github.com/shengliu66/ELR
        # criterion = elr_loss(num_examp=len(train_loader.dataset), num_classes=NUM_CLASS, _lambda=1, beta=0.9)
        # criterion = elr_plus_loss(num_examp=len(train_loader.dataset), num_classes=5, _lambda=1, beta=0.9)
        logger.info(criterion.__str__())

        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        # https://github.com/clovaai/AdamP
        from adamp import AdamP
        optimizer = AdamP(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                          weight_decay=args.weight_decay)

        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
        #                               weight_decay=args.weight_decay)
        optimizer = Lookahead(optimizer)

        if args.cuda:
            model.cuda()
            model = nn.DataParallel(model)
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
                # checkpoint['state_dict'].pop('module.pretrained.fc.bias', None)
                # checkpoint['state_dict'].pop('module.pretrained.fc.weight', None)
                # checkpoint['state_dict'].pop('module.head.1.weight', None)
                # checkpoint['state_dict'].pop('module.head.1.bias', None)
                # checkpoint['state_dict'].pop('module.head2.2.weight', None)
                # checkpoint['state_dict'].pop('module.head2.2.bias', None)
                # checkpoint['state_dict']['module.pretrained.fc.bias'] = torch.randn([5], dtype=torch.float64)
                # checkpoint['state_dict']['module.pretrained.fc.weight'] = torch.randn([5, 2048], dtype=torch.float64)
                # checkpoint['state_dict']['module.head.1.weight'] = torch.randn([5, 2048], dtype=torch.float64)
                # checkpoint['state_dict']['module.head.1.bias'] = torch.randn([5], dtype=torch.float64)
                # checkpoint['state_dict']['module.head2.2.weight'] = torch.randn([5, 4096], dtype=torch.float64)
                # checkpoint['state_dict']['module.head2.2.bias'] = torch.randn([5], dtype=torch.float64)

                lst = ['module.pretrained.fc.weight', 'module.pretrained.fc.bias', 'module.head.1.weight',
                       'module.head.1.bias', 'module.head2.2.weight', 'module.head2.2.bias']
                pretrained_dict = checkpoint['state_dict']
                new_model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in lst}
                new_model_dict.update(pretrained_dict)
                model.load_state_dict(new_model_dict, strict=False)
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

        scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs,
                                 len(train_loader) // args.batch_size,
                                 args.lr_step, warmup_epochs=4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
        #                                                                  T_0=10, T_mult=1,
        #                                                                  eta_min=1e-4,
        #                                                                  last_epoch=-1)

        # if args.eval:
        #     validate(args.start_epoch)
        #     return

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
