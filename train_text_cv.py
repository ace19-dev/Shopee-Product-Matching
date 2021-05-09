'''
https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205
https://modulabs-biomedical.github.io/Bias_vs_Variance

-> low bias, high variance
'''

import pprint
import timeit
from tqdm import tqdm

from transformers import BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel
from transformers import get_linear_schedule_with_warmup

from tensorboardX import SummaryWriter
# from torch.autograd import Variable

# import transforms
from models import text_model as M
from datasets.product_text import ProductTextDataset, NUM_CLASS
from datasets.sampler import ImbalancedDatasetSampler
from option import Options
from training.losses import FocalLoss
from training.lr_scheduler import *
from training.taylor_cross_entropy_loss import TaylorCrossEntropyLoss
from utils.training_helper import *
from utils.image_helper import *

# global variable
best_pred = 0.0
acc_lst_train = []
acc_lst_val = []


def main():
    global best_pred, acc_lst_train, acc_lst_val, lr

    args, args_desc = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    scheduler_params = {
        "lr_start": 1e-5,  # 2e-5
        "lr_max": 1e-5 * args.batch_size/2,
        # "lr_min": 2e-6,  # 2e-6
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }

    logger, log_file, final_output_dir, tb_log_dir, create_at = create_logger(args, args_desc)
    logger.info('-------------- params --------------')
    logger.info(pprint.pformat(args.__dict__))
    logger.info('-------------- scheduler_params --------------')
    logger.info(pprint.pformat(scheduler_params))


    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    npz_files = os.listdir(os.path.join(args.dataset_root, 'fold'))
    npz_files.sort()
    npz_files = npz_files[1:]
    num = int(len(npz_files) / 2)
    train_npzs = npz_files[:num]
    val_npzs = npz_files[num:]

    def train(epoch):
        global best_pred, acc_lst_train, acc_lst_val

        local_step = 0

        loss_score = AverageMeter()
        acc_score = AverageMeter()

        model.train()

        # last_time = time.time()
        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, batch in enumerate(tbar):
            # scheduler(optimizer, batch_idx, epoch, best_pred)

            batch = {k: v.cuda() for k, v in batch.items()}
            # if args.cuda:
            #     input_ids, input_mask, labels = \
            #         input_ids.cuda(), input_mask.cuda(), labels.cuda()

            local_step += 1

            # ArcFace
            preds = model(batch)
            loss = criterion(preds, batch['labels'])
            # loss = criterion(activations=outputs,
            #                  labels=torch.nn.functional.one_hot(targets),
            #                  t1=0.5, t2=1.5)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()

            batch_size = batch['input_ids'].size(0)
            loss_score.update(loss.item(), batch_size)
            preds = preds.argmax(dim=1)
            correct = (preds == batch['labels']).float().mean()
            acc_score.update(correct.item(), batch_size)

            # if batch_idx % 10 == 0:
            #     tbar.set_description('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
            #         epoch, batch_idx * len(input_ids), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss_score.avg, acc_score.avg))
            tbar.set_postfix(train_loss=loss_score.avg, accuracy=acc_score.avg, lr=optimizer.param_groups[0]['lr'])

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
        for batch in tbar:
            batch = {k: v.cuda() for k, v in batch.items()}
            # if args.cuda:
            #     input_ids, input_mask, labels = \
            #         input_ids.cuda(), input_mask.cuda(), labels.cuda()

            with torch.no_grad():
                # ArcFace
                preds = model(batch)
                loss = criterion(preds, batch['labels'])
                # test_loss += criterion(activations=outputs,
                #                        labels=torch.nn.functional.one_hot(targets),
                #                        t1=0.5, t2=1.5)
                batch_size = batch['input_ids'].size(0)
                loss_score.update(loss.item(), batch_size)
                preds = preds.argmax(dim=1)
                correct = (preds == batch['labels']).float().mean()
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
            create_at=create_at, filename=args.checkpoint_name, foldname=valset.fold_name())

    logger.info('\n-------------- tokenizer --------------')
    if args.model == 'DistilBERT':
        model_name = 'cahya/distilbert-base-indonesian'
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        bert_model = DistilBertModel.from_pretrained(model_name)
    elif args.model == 'roberta':
        model_name = 'cahya/roberta-base-indonesian-522M'
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        bert_model = RobertaModel.from_pretrained(model_name)
    logger.info(tokenizer)
    logger.info('\n')

    # for train_filename, val_filename in zip(train_npzs, val_npzs):
    for train_filename, val_filename in zip(['train_all_34250.npy'], ['valid_fold0_6850.npy']):
        logger.info('*****************************************')
        logger.info('fold: %s' % (train_filename.split('_')[1]))
        logger.info('train filename: %s' % (train_filename))
        logger.info('val filename: %s' % (val_filename))
        logger.info('*****************************************\n')

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

        trainset = ProductTextDataset(data_dir=args.dataset_root,
                                      fold=[train_filename],
                                      csv=['train.csv'],
                                      mode='train',
                                      tokenizer=tokenizer)
        valset = ProductTextDataset(data_dir=args.dataset_root,
                                    fold=[val_filename],
                                    csv=['train.csv'],
                                    mode='val',
                                    tokenizer=tokenizer)

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

        model = M.Model(bert_model, num_classes=NUM_CLASS)
        logger.info('\n-------------- model details --------------')
        logger.info(model)

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
        # logger.info('\n-------------- loss details --------------')
        # logger.info(criterion.__str__())

        logger.info('\n-------------- optimizer details --------------')
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=args.lr,
        #                             momentum=args.momentum, weight_decay=args.weight_decay)
        # https://github.com/clovaai/AdamP
        from adamp import AdamP
        optimizer = AdamP(model.parameters(), lr=scheduler_params['lr_start'],
                          betas=(0.9, 0.999), weight_decay=args.weight_decay)
        # optimizer = AdamP(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
        #                   weight_decay=args.weight_decay)
        logger.info(optimizer.__str__())

        # optimizer = Lookahead(optimizer)
        # logger.info(optimizer.__str__())

        logger.info('\n-------------- scheduler details --------------')
        # scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs,
        #                          len(train_loader) // args.batch_size,
        #                          args.lr_step, warmup_epochs=4)
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
                # lst = ['module.pretrained.fc.weight', 'module.pretrained.fc.bias', 'module.head.1.weight',
                #        'module.head.1.bias', 'module.head2.2.weight', 'module.head2.2.bias']
                # pretrained_dict = checkpoint['state_dict']
                # new_model_dict = model.state_dict()
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in lst}
                # new_model_dict.update(pretrained_dict)
                # model.load_state_dict(new_model_dict, strict=False)
                model.load_state_dict(checkpoint['state_dict'], strict=False)

                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                raise RuntimeError("=> no resume checkpoint found at '{}'".format(args.resume))

        start = timeit.default_timer()
        for epoch in range(args.start_epoch, args.epochs + 1):
            logger.info('\n\n[%s] ------- Epoch %d -------' % (time.strftime("%Y/%m/%d %H:%M:%S"), epoch))
            train(epoch)
            validate(epoch)

        if tokenizer is not None:
            directory = "%s/%s/%s/" % (args.output, 'shopee-product-matching', args.model)
            tokenizer.save_pretrained(directory + '(' + create_at + ')tokenizer')

        end = timeit.default_timer()
        logger.info('trained minute:%d' % (int((end - start) / 60)))
        logger.info('%s, training done.\n' % (train_filename.split('_')[1]))
        # logger.info('-------------- Inference Result --------------\n')

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
