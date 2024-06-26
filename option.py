import argparse


class Options():
    def __init__(self):
        # Training settings
        parser = argparse.ArgumentParser(description='shopee-product-matching')
        parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
        parser.add_argument('--tb-log', type=str,
                            default='tb_log',
                            help='log directory name')
        parser.add_argument('--dataset-root', type=str,
                            default='/home/ace19/dl_data/shopee-product-matching',
                            help='root')
        parser.add_argument('--output', type=str,
                            default='experiments',
                            help='output directory name')
        parser.add_argument('--dataset', type=str, default='product',
                            help='training dataset')
        # parser.add_argument('--model', type=str, default='tf_efficientnet_b4_ns',
        #                     help='network model type (default: efficientnet_b3)')
        parser.add_argument('--model', type=str, default='xlm-roberta-base',
                            help='network model type')
        parser.add_argument('--pretrained', action='store_true',
                            default=False, help='load pretrianed mode')
        # parser.add_argument('--widen', type=int, default=4, metavar='N',
        #     help='widen factor of the network (default: 4)')
        # training hyper params
        parser.add_argument('--batch-size', type=int, default=64,
                            metavar='N', help='batch size for training (default: 128)')
        parser.add_argument('--test-batch-size', type=int, default=64,
                            metavar='N', help='batch size for testing (default: 256)')
        parser.add_argument('--epochs', type=int, default=30, metavar='N',
                            help='number of epochs to train (default: 600)')
        parser.add_argument('--start_epoch', type=int, default=1,
                            metavar='N', help='the epoch number to start (default: 1)')
        parser.add_argument('--workers', type=int, default=4,
                            metavar='N', help='dataloader threads')
        # lr:
        # tf_efficientnet_b2_ns w/ cosine-softmax: 0.0002
        # tf_efficientnet_b2_ns w/ ArcFace: 0.00001
        # dm_nfnet_f0 w/ cosine-softmax: 0.0003
        # dm_nfnet_f0 w/ ArcFace: 0.00001
        # parser.add_argument('--lr', type=float, default=0.00002, metavar='LR',
        #                     help='learning rate (default: 0.1)')
        parser.add_argument('--lr', type=float, default=0.00002, metavar='LR',
                            help='learning rate (default: 0.1)')
        parser.add_argument('--lr-scheduler', type=str, default='cos',
                            help='learning rate scheduler (default: cos)')
        parser.add_argument('--lr-step', type=int, default=5, metavar='LR',
                            help='learning rate step (default: 40)')
        # optimizer
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='SGD momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=0.01,
                            metavar='M', help='SGD weight decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true',
                            default=False, help='disables CUDA training')
        # parser.add_argument('--plot', action='store_true', default=False,
        #     help='matplotlib')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        # checking point
        parser.add_argument('--resume', type=str,
                            default=None,
                            # default='pre-trained/noisy-student-efficientnet-b2.pth',
                            # 'efficientnet-b7_ns_aa-original-mstd0.5_re_100k_v4_cad79a/snapshot_100000.fp16.pth',
                            # default='experiments/shopee-product-matching/tf_efficientnet_b4_ns/'
                            #         '(2021-05-08_01:16:34)product_fold3_512x512_tf_efficientnet_b4_ns_acc(80.22773)_loss(2.70676)_checkpoint11.pth.tar',
                            help='put the path to resuming file if needed')
        parser.add_argument('--checkpoint_name', type=str, default='product',
                            help='set the checkpoint name')
        # parser.add_argument('--tsne_result', type=str,
        #                     # default=None,
        #                     default='tsne_result',
        #                     help='directory to save .png file')
        parser.add_argument('--result', type=str,
                            # default=None,
                            default='submit',
                            help='directory to save .csv file')
        # Grad-CAM
        # parser.add_argument('--image-path', type=str,
        #                     default='/home/ace19/dl_data/deepfake-detection-challenge/face_datasets/validation',
        #                     help='Input image path')
        # parser.add_argument('--output-path', type=str,
        #                     default='/home/ace19/dl_data/deepfake-detection-challenge/face_datasets/grad-cam',
        #                     help='Input image path')
        parser.add_argument('--beta', default=0.5, type=float,
                            help='hyperparameter beta')
        # parser.add_argument('--cutmix_prob', default=0.5, type=float,
        #                     help='cutmix probability')
        parser.add_argument('--cutmix_prob', default=0.2, type=float,
                            help='cutmix probability')
        parser.add_argument('--alpha', default=0.8, type=float,
                            help='mixup interpolation coefficient (default: 1)')
        parser.add_argument('--mixup_prob', default=0.0, type=float,
                            help='mixup probability')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args, self.parser.description
