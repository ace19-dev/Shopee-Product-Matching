'''
https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205
https://modulabs-biomedical.github.io/Bias_vs_Variance

-> low bias, high variance
'''

import pprint
import timeit
from tqdm import tqdm

from transformers import BertTokenizer, AutoTokenizer
# from transformers import BertModel, BertConfig, AdamW
from transformers import get_linear_schedule_with_warmup

from tensorboardX import SummaryWriter
# from torch.autograd import Variable

# import transforms
from models import text_model as M
from datasets.product import ProductTextDataset, NUM_CLASS
from datasets.sampler import ImbalancedDatasetSampler
from option import Options
from training.losses import FocalLoss
from training.lr_scheduler import LR_Scheduler
from training.taylor_cross_entropy_loss import TaylorCrossEntropyLoss
from utils.training_helper import *
from utils.image_helper import *

# global variable
best_pred = 0.0
acc_lst_train = []
acc_lst_val = []
lr = 0.0


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

    # world_size = int(os.environ['WORLD_SIZE'])
    # rank = int(os.environ['RANK'])
    # dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    # dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    # local_rank = args.local_rank
    # torch.cuda.set_device(local_rank)

    logger, log_file, final_output_dir, tb_log_dir, create_at = create_logger(args, args_desc)
    logger.info('-------------- params --------------')
    logger.info(pprint.pformat(args.__dict__))

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

        losses = AverageMeter()
        accs = AverageMeter()

        model.train()

        # last_time = time.time()
        tbar = tqdm(train_loader, desc='\r')
        for batch_idx, (input_ids, input_mask, labels, pos_id) in enumerate(tbar):
            scheduler(optimizer, batch_idx, epoch, best_pred)

            if args.cuda:
                input_ids, input_mask, labels = \
                    input_ids.cuda(), input_mask.cuda(), labels.cuda()

            local_step += 1

            # ArcFace
            # _, outputs = model(input_ids, input_mask, labels)
            # # cosine-softmax
            # # https://huggingface.co/transformers/model_doc/bert.html
            _, outputs = model(input_ids, input_mask)

            # loss = criterion(activations=outputs,
            #                  labels=torch.nn.functional.one_hot(targets),
            #                  t1=0.5, t2=1.5)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs.data, 1)
            # print('preds:', preds)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()

            batch_size = float(input_ids.size(0))
            # https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types
            losses.update(loss.data.cpu().numpy().item(), batch_size)
            # https://discuss.pytorch.org/t/trying-to-pass-too-many-cpu-scalars-to-cuda-kernel/87757/4
            correct = torch.sum(preds == labels.data)
            accs.update(correct.cpu().numpy().item(), batch_size)

            if batch_idx % 10 == 0:
                tbar.set_description('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                    epoch, batch_idx * len(input_ids), len(train_loader.dataset),
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
        for batch_idx, (input_ids, input_mask, labels, pos_id) in enumerate(tbar):
            if args.cuda:
                input_ids, input_mask, labels = \
                    input_ids.cuda(), input_mask.cuda(), labels.cuda()

            with torch.no_grad():
                # ArcFace
                # _, outputs = model(input_ids, input_mask, labels)
                # # cosine-softmax
                _, outputs = model(input_ids, input_mask)

                # test_loss += criterion(activations=outputs,
                #                        labels=torch.nn.functional.one_hot(targets),
                #                        t1=0.5, t2=1.5)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs.data, 1)
                correct = torch.sum(preds == labels.data)

                batch_size = float(input_ids.size(0))
                losses.update(loss.data.cpu().numpy().item(), batch_size)
                accs.update(correct.cpu().numpy().item(), batch_size)

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
            'optimizer': optimizer.state_dict(),
            'best_pred': best_pred,
            'acc_lst_train': acc_lst_train,
            'acc_lst_val': acc_lst_val,
        }, logger=logger, args=args, loss=losses.avg, is_best=is_best,
            create_at=create_at, filename=args.checkpoint_name, foldname=valset.fold_name())

    logger.info('\n-------------- tokenizer --------------')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=False)
    logger.info(tokenizer)
    logger.info('\n')

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

        trainset = ProductTextDataset(data_dir=args.dataset_root,
                                      fold=[train_filename],
                                      csv=['train.csv'],
                                      mode='train',
                                      tokenizer=tokenizer)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(
        #     trainset, shuffle=True)
        valset = ProductTextDataset(data_dir=args.dataset_root,
                                    fold=[val_filename],
                                    csv=['train.csv'],
                                    mode='val',
                                    tokenizer=tokenizer)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(
        #     valset, shuffle=False)

        logger.info('\n-------------- dataset --------------')
        logger.info(trainset)
        logger.info(valset)

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

        # config = BertConfig.from_pretrained('bert-base-multilingual-cased')
        # model = BertModel(config)
        model = M.Model(n_classes=NUM_CLASS, model_name='bert', use_fc=True)
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
        optimizer = AdamP(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                          weight_decay=args.weight_decay)
        logger.info(optimizer.__str__())

        # optimizer = Lookahead(optimizer)
        # logger.info(optimizer.__str__())

        logger.info('\n-------------- scheduler details --------------')
        # scheduler = \
        #     torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, args.epochs)
        scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs,
                                 len(train_loader) // args.batch_size,
                                 args.lr_step, warmup_epochs=4)
        total_steps = len(train_loader) * args.epochs
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=(total_steps/args.epochs)*3,
        #                                             num_training_steps=total_steps)
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
