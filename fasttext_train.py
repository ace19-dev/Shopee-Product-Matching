import os
import pprint
import timeit
from tqdm import tqdm
import pandas as pd
from gensim.models import FastText

import torch.utils.data as data
from torch.nn import functional as F

from datasets.sampler import ImbalancedDatasetSampler
from datasets.product_text import WordVectorDataset
from option import Options
from training.losses import FocalLoss
from training.lr_scheduler import *
from utils.image_helper import *
from utils.training_helper import *

# global variable
best_pred = 0.0
acc_lst_train = []
acc_lst_val = []
lr = 0.0


class FTModel(nn.Module):
    def __init__(self, n_classes=11014, dropout=0.1):
        super(FTModel, self).__init__()

        self.fc = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.fc(x)

        features = x
        # Features in rows, normalize axis 1.
        features = F.normalize(features, p=2, dim=1, eps=1e-8)

        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return features, x


def main():
    global best_pred, acc_lst_train, acc_lst_val, lr

    args, args_desc = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    args.model = 'FastText'

    logger, log_file, final_output_dir, tb_log_dir, create_at = \
        create_logger(args, args_desc)
    logger.info('-------------- params --------------')
    logger.info(pprint.pformat(args.__dict__))

    # npz_files = os.listdir(os.path.join(args.dataset_root, 'fold'))
    # npz_files.sort()
    # npz_files = [f for f in npz_files if f[-3:] != 'log']
    # num = int(len(npz_files) / 2)
    # train_npzs = npz_files[:num]
    # val_npzs = npz_files[num:]

    def train(epoch):
        global best_pred, acc_lst_train, acc_lst_val

        loss_score = AverageMeter()
        acc_score = AverageMeter()

        model.train()

        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (features, targets, pos_id) in enumerate(tbar):
            scheduler(optimizer, batch_idx, epoch, best_pred)

            if args.cuda:
                features, targets = features.cuda(), targets.cuda()

            _, preds = model(features)
            loss = criterion(preds, targets)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            batch_size = features.size(0)
            loss_score.update(loss.item(), batch_size)
            preds = preds.argmax(dim=1)
            correct = (preds == targets).float().mean()
            acc_score.update(correct.item(), batch_size)

            # if batch_idx % 10 == 0:
            #     tbar.set_description('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
            #         epoch, batch_idx * len(images), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss_score.avg, acc_score.avg))
            tbar.set_postfix(Train_Loss=loss_score.avg, accuracy=acc_score.avg, LR=optimizer.param_groups[0]['lr'])

        logger.info('[Train] Loss: {:.4f} Acc: {:.4f}'.format(loss_score.avg, acc_score.avg))

    def validate(epoch):
        global best_pred, acc_lst_train, acc_lst_val
        is_best = False

        loss_score = AverageMeter()
        acc_score = AverageMeter()

        model.eval()

        tbar = tqdm(val_loader, desc='\r')
        for batch_idx, (features, targets, pos_id) in enumerate(tbar):
            if args.cuda:
                features, targets = features.cuda(), targets.cuda()

            with torch.no_grad():
                _, preds = model(features)
                loss = criterion(preds, targets)

                batch_size = features.size(0)
                loss_score.update(loss.item(), batch_size)
                preds = preds.argmax(dim=1)
                correct = (preds == targets).float().mean()
                acc_score.update(correct.item(), batch_size)

                # tbar.set_description('\r[Validate] Loss: %.5f | Top1: %.5f' % (loss_score.avg, acc_score.avg))
                tbar.set_postfix(valid_loss=loss_score.avg, accuracy=acc_score.avg)

        logger.info('[Validate] Loss: %.5f | Acc: %.5f' % (loss_score.avg, acc_score.avg))

        # save checkpoint
        acc_lst_val += [acc_score.avg]
        if acc_score.avg > best_pred:
            logger.info('** [best_pred]: {:.4f}'.format(acc_score.avg))
            best_pred = acc_score.avg
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_pred': best_pred,
            'acc_lst_train': acc_lst_train,
            'acc_lst_val': acc_lst_val,
        }, logger=logger, args=args, loss=loss_score.avg, is_best=is_best,
            create_at=create_at, filename=args.checkpoint_name, foldname='ALL')

    # for train_filename, val_filename in zip(train_npzs, val_npzs):
    for train_filename, val_filename in zip(['train_all_34250.npy'], ['train_all_34250.npy']):
        # logger.info('****************************')
        # logger.info('fold: %s' % (train_filename.split('_')[1]))
        # logger.info('train filename: %s' % (train_filename))
        # logger.info('val filename: %s' % (val_filename))
        # logger.info('****************************\n')

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

        fasttext_model = FastText.load('results/scratch_fasttext.model')

        trainset = WordVectorDataset(data_dir=args.dataset_root,
                                     csv='train.csv',
                                     file='corpus.txt',
                                     model=fasttext_model,
                                     mode='train', )
        valset = WordVectorDataset(data_dir=args.dataset_root,
                                   csv='train.csv',
                                   file='corpus.txt',
                                   model=fasttext_model,
                                   mode='val', )

        logger.info('\n-------------- dataset --------------')
        logger.info(trainset)
        logger.info(valset)

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

        model = FTModel(dropout=0.1)
        # model.half()  # to save space.
        logger.info('\n-------------- model details --------------')
        print(model)

        # freeze_until(model, "pretrained.blocks.5.0.conv_pw.weight")
        # keys = [k for k, v in model.named_parameters() if v.requires_grad]
        # print(keys)
        # freeze_bn(model)

        # criterion and optimizer
        # https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py
        # criterion = TaylorCrossEntropyLoss(n=6, ignore_index=255, reduction='mean',
        #                                    num_cls=NUM_CLASS, smoothing=0.0)
        # criterion = torch.nn.CrossEntropyLoss()
        criterion = FocalLoss()

        logger.info('\n-------------- loss details --------------')
        logger.info(criterion.__str__())

        logger.info('\n-------------- optimizer details --------------')
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=args.lr,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)
        # https://github.com/clovaai/AdamP
        from adamp import AdamP
        optimizer = AdamP(model.parameters(), lr=args.lr,
                          betas=(0.9, 0.999), weight_decay=args.weight_decay)
        logger.info(optimizer.__str__())

        logger.info('\n-------------- scheduler details --------------')
        scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs,
                                 len(train_loader) // args.batch_size,
                                 args.lr_step, warmup_epochs=0)
        logger.info(scheduler.__str__())

        if args.cuda:
            model.cuda()
            model = nn.DataParallel(model)
            criterion.cuda()

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
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                raise RuntimeError("=> no resume checkpoint found at '{}'".format(args.resume))

        start = timeit.default_timer()
        for epoch in range(args.start_epoch, args.epochs + 1):
            logger.info('\n\n[%s] ------- Epoch %d -------' % (time.strftime("%Y/%m/%d %H:%M:%S"), epoch))
            train(epoch)
            validate(epoch)

        end = timeit.default_timer()
        logger.info('trained time:%d' % (int((end - start) / 60)))
        logger.info('%s, training done.\n' % (train_filename.split('_')[1]))


if __name__ == '__main__':
    main()
