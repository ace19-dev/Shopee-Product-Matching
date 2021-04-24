'''
https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205
https://modulabs-biomedical.github.io/Bias_vs_Variance

-> low bias, high variance

1st place solution: https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/221957
Our final submission first averaged the probabilities of the predicted classes of ViT and ResNext.
This averaged probability vector was then merged with the predicted probabilities of EfficientnetB4 and CropNet
in the second stage. For this purpose, the values were simply summed up -> 확율 누적.

https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images

'''

import pprint
import timeit
from tqdm import tqdm

from tensorboardX import SummaryWriter
from torch.autograd import Variable

import transforms
from models import model as M
from models.shopeenet import ShopeeNet
from datasets.product import ProductDataset, NUM_CLASS
from datasets.sampler import ImbalancedDatasetSampler
from option import Options
from training.lr_scheduler import *
from training.losses import FocalLoss
from training.taylor_cross_entropy_loss import TaylorCrossEntropyLoss
from utils.training_helper import *
from utils.image_helper import *

# global variable
best_pred = 0.0
acc_lst_train = []
acc_lst_val = []
lr = 0.0

IMAGE_SIZE = str(transforms.CROP_HEIGHT) + 'x' + \
             str(transforms.CROP_WIDTH)


# SCHEDULER = 'CosineAnnealingWarmRestarts'  # 'CosineAnnealingLR'
# factor = 0.2  # ReduceLROnPlateau
# patience = 4  # ReduceLROnPlateau
# eps = 1e-6  # ReduceLROnPlateau
# T_max = 10  # CosineAnnealingLR
# T_0 = 4  # CosineAnnealingWarmRestarts
# min_lr = 1e-6
#
#
# def fetch_scheduler(SCHEDULER, optimizer):
#     if SCHEDULER == 'ReduceLROnPlateau':
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True,
#                                       eps=eps)
#     elif SCHEDULER == 'CosineAnnealingLR':
#         scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=min_lr, last_epoch=-1)
#     elif SCHEDULER == 'CosineAnnealingWarmRestarts':
#         scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=min_lr, last_epoch=-1)
#     return scheduler


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

    scheduler_params = {
        "lr_start": 1e-5,   # 2e-5
        "lr_max": 1e-5 * args.batch_size,
        "lr_min": 1e-6,     # 2e-6
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }

    # world_size = int(os.environ['WORLD_SIZE'])
    # rank = int(os.environ['RANK'])
    # dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    # dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    # local_rank = args.local_rank
    # torch.cuda.set_device(local_rank)

    logger, log_file, final_output_dir, tb_log_dir, create_at = \
        create_logger(args, args_desc, IMAGE_SIZE)
    logger.info('-------------- params --------------')
    logger.info(pprint.pformat(args.__dict__))

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    npz_files = os.listdir(os.path.join(args.dataset_root, 'fold'))
    npz_files.sort()
    npz_files = [f for f in npz_files if f[-3:] != 'log']
    num = int(len(npz_files) / 2)
    train_npzs = npz_files[:num]
    val_npzs = npz_files[num:]

    def train(epoch):
        global best_pred, acc_lst_train, acc_lst_val

        local_step = 0
        MIXUP_FLAG = False

        loss_score = AverageMeter()
        acc_score = AverageMeter()

        model.train()

        # last_time = time.time()
        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (images, targets, fnames) in enumerate(tbar):
            # scheduler(optimizer, batch_idx, epoch, best_pred)

            if args.cuda:
                images, targets = images.cuda(), targets.cuda()

            local_step += 1

            # TODO: move cutmix/fmix/mixup func to ProductDataset .py
            # https://www.kaggle.com/sebastiangnther/cassava-leaf-disease-vit-tpu-training
            # CutMix (from https://github.com/clovaai/CutMix-PyTorch)
            r = np.random.rand(1)
            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = targets
                target_b = targets[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))

                _, outputs = model(images, targets)
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

                    _, outputs = model(images, targets)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    # train_loss += loss.data[0]
                    # _, preds = torch.max(outputs.data, 1)
                    # total += targets.size(0)
                    # correct += (lam * preds.eq(targets_a.data).cpu().sum().float()
                    #             + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float())
                else:
                    MIXUP_FLAG = False
                    _, outputs = model(images, targets)
                    # print('outputs:', outputs.shape)
                    # print('targets:', targets.shape)

                    # loss = criterion(activations=outputs,
                    #                  labels=torch.nn.functional.one_hot(targets),
                    #                  t1=0.5, t2=1.5)
                    loss = criterion(outputs, targets)

            preds = torch.argmax(outputs.data, 1)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            batch_size = float(images.size(0))
            # https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types
            loss_score.update(loss.data.cpu().numpy().item(), batch_size)
            if MIXUP_FLAG:
                # TODO: accuracy bugfix
                top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() +
                        (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()).long()
                acc_score.update(top1.cpu().numpy().item(), batch_size)

            else:
                # https://discuss.pytorch.org/t/trying-to-pass-too-many-cpu-scalars-to-cuda-kernel/87757/4
                top1 = accuracy(outputs, targets)[0]
                # top1 = torch.sum(preds == targets.data)
                acc_score.update(top1.cpu().numpy().item(), batch_size)

            # if batch_idx % 10 == 0:
            #     tbar.set_description('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
            #         epoch, batch_idx * len(images), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss_score.avg, acc_score.avg))
            tbar.set_postfix(Train_Loss=loss_score.avg, Epoch=epoch, LR=optimizer.param_groups[0]['lr'])

        scheduler.step()

        logger.info('[Train] Loss: {:.4f} Acc: {:.4f}'.format(loss_score.avg, acc_score.avg))

        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss_score.avg, global_steps)
        writer.add_scalar('train_acc', acc_score.avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    def validate(epoch):
        global best_pred, acc_lst_train, acc_lst_val
        is_best = False

        loss_score = AverageMeter()
        acc_score = AverageMeter()

        model.eval()

        tbar = tqdm(val_loader, desc='\r')
        for batch_idx, (images, targets, fnames) in enumerate(tbar):
            if args.cuda:
                images, targets = images.cuda(), targets.cuda()

            with torch.no_grad():
                _, outputs = model(images, targets)

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
                top1 = accuracy(outputs, targets)[0]
                batch_size = float(images.size(0))
                loss_score.update(loss.data.cpu().numpy().item(), batch_size)
                acc_score.update(top1.cpu().numpy().item(), batch_size)

                # tbar.set_description('\r[Validate] Loss: %.5f | Top1: %.5f' % (loss_score.avg, acc_score.avg))
                tbar.set_postfix(Eval_Loss=loss_score.avg)

        logger.info('[Validate] Loss: %.5f | Acc: %.5f' % (loss_score.avg, acc_score.avg))

        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', loss_score.avg, global_steps)
        writer.add_scalar('valid_acc', acc_score.avg, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

        # save checkpoint
        acc_lst_val += [acc_score.avg]
        if acc_score.avg > best_pred:
            logger.info('** [best_pred]: {:.4f}'.format(acc_score.avg))
            best_pred = acc_score.avg
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acc_lst_train': acc_lst_train,
            'acc_lst_val': acc_lst_val,
        }, logger=logger, args=args, loss=loss_score.avg, is_best=is_best,
            image_size=IMAGE_SIZE, create_at=create_at, filename=args.checkpoint_name,
            foldname=valset.fold_name())

    for train_filename, val_filename in zip(train_npzs, val_npzs):
        logger.info('****************************')
        logger.info('fold: %s' % (train_filename.split('_')[1]))
        logger.info('train filename: %s' % (train_filename))
        logger.info('val filename: %s' % (val_filename))
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
                                  transform=transforms.training_augmentation3(), )
        # train_sampler = torch.utils.data.distributed.DistributedSampler(
        #     trainset, shuffle=True)
        valset = ProductDataset(data_dir=args.dataset_root,
                                fold=[val_filename],
                                csv=['train.csv'],
                                mode='val',
                                transform=transforms.validation_augmentation())
        # val_sampler = torch.utils.data.distributed.DistributedSampler(
        #     valset, shuffle=False)

        logger.info('-------------- train transform --------------')
        logger.info(transforms.training_augmentation3())
        logger.info('\n-------------- valid transform --------------')
        logger.info(transforms.validation_augmentation())

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
                                                   # sampler=train_sampler,
                                                   pin_memory=True,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(valset,
                                                 batch_size=args.batch_size,
                                                 # sampler=val_sampler,
                                                 shuffle=False,
                                                 num_workers=args.workers,
                                                 pin_memory=True)

        # cosine-softmax
        # model = M.Model(model_name=args.model, nclass=NUM_CLASS)
        # https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images
        model = ShopeeNet(n_classes=NUM_CLASS, model_name=args.model,
                          use_fc=True, fc_dim=512, dropout=0.1)
        # model.half()  # to save space.
        logger.info('\n-------------- model details --------------')
        print(model)

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
        #                                    num_cls=NUM_CLASS, smoothing=0.0)
        # criterion = torch.nn.CrossEntropyLoss()
        # criterion = LabelSmoothingLoss(NUM_CLASS, smoothing=0.1)
        criterion = FocalLoss()
        # https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271
        # criterion = FocalCosineLoss()
        logger.info('\n-------------- loss details --------------')
        logger.info(criterion.__str__())

        logger.info('\n-------------- optimizer details --------------')
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        # https://github.com/clovaai/AdamP
        from adamp import AdamP
        optimizer = AdamP(model.parameters(), lr=scheduler_params['lr_start'],
                          betas=(0.9, 0.999), weight_decay=args.weight_decay)
        # optimizer = torch.optim.Adam(model.parameters(), lr=scheduler_params['lr_start'],
        #                              betas=(0.9, 0.999), weight_decay=args.weight_decay)
        logger.info(optimizer.__str__())

        # optimizer = Lookahead(optimizer)
        # logger.info(optimizer.__str__())

        logger.info('\n-------------- scheduler details --------------')
        # scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs,
        #                          len(train_loader) // args.batch_size,
        #                          args.lr_step, warmup_epochs=4)
        # https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images
        scheduler = ShopeeScheduler(optimizer, **scheduler_params)
        logger.info(scheduler.__str__())

        if args.cuda:
            model.cuda()
            model = nn.DataParallel(model)
            criterion.cuda()

        # for ps in model.parameters():
        #     dist.broadcast(ps, 0)
        # model = torch.nn.parallel.DistributedDataParallel(
        #     module=model, broadcast_buffers=False, device_ids=[local_rank])

        # get the number of model parameters
        logger.info('\nNumber of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])) + '\n')

        # check point
        if args.resume is not None:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch'] + 1
                best_pred = checkpoint['best_pred']
                acc_lst_train = checkpoint['acc_lst_train']
                acc_lst_val = checkpoint['acc_lst_val']
                not_to_be_applied = ['module.backbone.classifier.weight',
                                     'module.backbone.classifier.bias',
                                     'module.cosine_softmax.weights',
                                     'module.cosine_softmax.scale',
                                     'module.cosine_softmax.fc.weight',
                                     'module.cosine_softmax.fc.bias'
                                     ]
                pretrained_dict = checkpoint['state_dict']
                new_model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in not_to_be_applied}
                new_model_dict.update(pretrained_dict)
                model.load_state_dict(new_model_dict, strict=False)

                # model.load_state_dict(checkpoint['state_dict'], strict=False)

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

        # dist.destroy_process_group()

        end = timeit.default_timer()
        logger.info('trained time:%d' % (int((end - start) / 3600)))
        logger.info('%s, training done.\n' % (train_filename.split('_')[1]))
        # logger.info('-------------- Inference Result --------------\n')

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
