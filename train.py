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


def getMetric(col):
    def f1score(row):
        n = len(np.intersect1d(row.target, row[col]))
        return 2 * n / (len(row.target) + len(row[col]))

    return f1score


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

    best_pred = 0.0

    args, args_desc = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    scheduler_params = {
        "lr_start": 2e-5,  # 2e-5
        "lr_max": 2e-5 * args.batch_size,
        # "lr_max": 2e-6,
        # "lr_min": 1e-7,     # 2e-6
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }

    logger, log_file, final_output_dir, tb_log_dir, create_at = \
        create_logger(args, args_desc, IMAGE_SIZE)
    logger.info('-------------- params --------------')
    logger.info(pprint.pformat(args.__dict__))

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

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    trainset = ProductDataset(data_dir=args.dataset_root,
                              fold=['train_all_34250.npy'],
                              csv=['train.csv'],
                              mode='train',
                              transform=transforms.training_augmentation3(), )
    valset = ProductDataset(data_dir=args.dataset_root,
                            fold=['train_all_34250.npy'],
                            csv=['train.csv'],
                            mode='val',
                            transform=transforms.validation_augmentation())

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
                                               pin_memory=True,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    # cosine-softmax
    # model = M.Model(model_name=args.model, nclass=NUM_CLASS)
    # https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images
    model = ShopeeNet(n_classes=NUM_CLASS, model_name=args.model,
                      use_fc=True, fc_dim=512, dropout=0.2)
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

            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            raise RuntimeError("=> no resume checkpoint found at '{}'".format(args.resume))

    def train(epoch):
        global best_pred, acc_lst_train, acc_lst_val

        local_step = 0

        loss_score = AverageMeter()
        acc_score = AverageMeter()

        model.train()

        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (images, targets, fnames) in enumerate(tbar):
            # scheduler(optimizer, batch_idx, epoch, best_pred)

            if args.cuda:
                images, targets = images.cuda(), targets.cuda()

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

                _, preds = model(images, targets)
                # loss = criterion(activations=outputs,
                #                  labels=torch.nn.functional.one_hot(target_a),
                #                  t1=0.5, t2=1.5) * lam + \
                #        criterion(activations=outputs,
                #                  labels=torch.nn.functional.one_hot(target_b),
                #                  t1=0.5, t2=1.5) * (1. - lam)
                loss = criterion(preds, target_a) * lam + \
                       criterion(preds, target_b) * (1. - lam)
            else:
                _, preds = model(images, targets)
                # loss = criterion(activations=outputs,
                #                  labels=torch.nn.functional.one_hot(targets),
                #                  t1=0.5, t2=1.5)
                loss = criterion(preds, targets)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            batch_size = images.size(0)
            loss_score.update(loss.item(), batch_size)
            preds = preds.argmax(dim=1)
            correct = (preds == targets).float().mean()
            acc_score.update(correct.item(), batch_size)

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
                _, preds = model(images, targets)
                loss = criterion(preds, targets)
                # test_loss += criterion(activations=outputs,
                #                        labels=torch.nn.functional.one_hot(targets),
                #                        t1=0.5, t2=1.5)
                batch_size = images.size(0)
                loss_score.update(loss.item(), batch_size)
                preds = preds.argmax(dim=1)
                correct = (preds == targets).float().mean()
                acc_score.update(correct.item(), batch_size)

                # tbar.set_description('\r[Validate] Loss: %.5f | Top1: %.5f' % (loss_score.avg, acc_score.avg))
                tbar.set_postfix(valid_loss=loss_score.avg, accuracy=acc_score.avg)

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

    start = timeit.default_timer()
    for epoch in range(args.start_epoch, args.epochs + 1):
        logger.info('\n\n[%s] ------- Epoch %d -------' % (time.strftime("%Y/%m/%d %H:%M:%S"), epoch))
        train(epoch)
        validate(epoch)

    # dist.destroy_process_group()

    end = timeit.default_timer()
    logger.info('trained minutes:%d' % (int((end - start) / 60)))
    # logger.info('-------------- Inference Result --------------\n')

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
