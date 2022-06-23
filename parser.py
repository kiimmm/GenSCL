import argparse

def str2bool(v):
    """
    Parse boolean using argument parser.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def genscl_parser():
    parser = argparse.ArgumentParser('argument for supervised contrastive learning')
    
    parser.add_argument('--desc', type=str, default=None,
                        help='experiment name')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model kind')
    parser.add_argument('--print-freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save-freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--save-root', type=str, default='./saves/',
                        help='root directory of results')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for generalized supervised contrastive loss')
    parser.add_argument('--resume', type=str, default=None)
    # knowledge distillation
    parser.add_argument('--KD', action='store_true', default=False,
                        help='perform knowledge distillation')
    parser.add_argument('--KD-alpha', type=float, default=1,
                        help='weight of KD')
    parser.add_argument('--KD-temp', type=float, default=1,
                        help='softening prediction of teachers')
    parser.add_argument('--teacher-kind', type=str, default='efficientnetv2_rw_m')
    parser.add_argument('--teacher-path', type=str, default=None)
    parser.add_argument('--teacher-ckpt', type=str, default='ckpt_best.pth')
    # optimization
    parser.add_argument('--optim-kind', type=str, default='SGD',
                        choices=['SGD', 'RMSProp', 'Adam', 'AdamW'],
                        help='kind of optimizer')
    parser.add_argument('--LARS', action='store_true', default=False)
    parser.add_argument('--learning-rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--cosine', type=str2bool, default=True,
                        help='using cosine annealing')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # warmup
    parser.add_argument('--warm', type=str2bool, default=True,
                        help='warm-up for large batch training')
    parser.add_argument('--warmup-from', type=float, default=0.01)
    parser.add_argument('--warm-epochs', type=int, default=10)
    parser.add_argument('--multiprocessing-distributed', action='store_true', default=False)
    # model dataset
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet'])
    # data augment policy
    parser.add_argument('--augment-policy', type=str, default='sim',
                        choices=['no', 'sim', 'auto', 'rand'],
                        help='data augmentation policy')
    # random augment
    parser.add_argument('--rand-n', type=int, default=1,
                        help='# of random augment')
    parser.add_argument('--rand-m', type=int, default=2,
                        help='magnitude of random augment')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='perform cutout')
    parser.add_argument('--n-holes', type=int, default=1,
                        help='# of cutout holes')
    parser.add_argument('--cutout-length', type=int, default=16,
                        help='length of a cutout hole')
    parser.add_argument('--mix', type=str, default=None,
                        choices=['mixup', 'cutmix', 'mixup_cutmix'],
                        help='image-based regularizations')
    parser.add_argument('--mix-alpha', type=float, default=1.0,
                        help='alpha for mixup/cutmix beta distribution')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug: train 1 epoch')
    # wandb
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='use wandb for visualization')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='your wandb id')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='wandb project name')
    return parser

def linear_parser():
    parser = argparse.ArgumentParser('argument for linear finetuning')
    
    parser.add_argument('--desc', type=str, default=None,
                        help='experiment name')
    parser.add_argument('--model', type=str, default='resnet50',
                        help='model kind')
    # model config
    parser.add_argument('--pretrained', type=str,
                        help='pretraiend encoder to load')
    parser.add_argument('--pretrained-ckpt', type=str, default='ckpt_last.pth',
                        help='pretrained encoder checkpoint')
    parser.add_argument('--label-smoothing', type=float, default=0.,
                        help='label smoothing for cross-entropy loss')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save-freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--save-root', type=str, default='./saves/',
                        help='root directory of results')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    
    # optimization
    parser.add_argument('--optim-kind', type=str, default='SGD',
                        choices=['SGD', 'RMSProp', 'Adam', 'AdamW'],
                        help='kind of optimizer')
    parser.add_argument('--LARS', action='store_true', default=False)
    parser.add_argument('--learning-rate', type=float, default=5,
                        help='learning rate')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[60,75,90],
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use automatic mixed precision')
    parser.add_argument('--multiprocessing-distributed', action='store_true', default=False)
    # warmup
    parser.add_argument('--warm', action="store_true", default=True,
                        help='warm-up for large batch training')
    parser.add_argument('--warmup-from', type=float, default=1e-5)
    parser.add_argument('--warm-epochs', type=int, default=5)
    # model dataset
    parser.add_argument('--dataset', type=str,
                        choices=['cifar10', 'cifar100', 'imagenet'])
    # data augment policy
    parser.add_argument('--augment-policy', type=str, default='sim',
                        choices=['no', 'sim', 'auto', 'rand'],
                        help='data augmentation policy')
    # random augment
    parser.add_argument('--rand-n', type=int, default=1,
                        help='# of random augment')
    parser.add_argument('--rand-m', type=int, default=2,
                        help='magnitude of random augment')
    # erasing
    parser.add_argument('--erasing', action='store_true', default=False,
                        help='perform erasing regularization')
    parser.add_argument('--erasing-p', type=float, default=0.5,
                        help='erasing probability')
    # cutout
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='perform cutout')
    parser.add_argument('--n-holes', type=int, default=1,
                        help='# of cutout holes')
    parser.add_argument('--cutout-length', type=int, default=16,
                        help='length of a cutout hole')
    parser.add_argument('--mix', type=str, default=None,
                        choices=['mixup', 'cutmix', 'mixup_cutmix'],
                        help='image-based regularizations')
    parser.add_argument('--mix-alpha', type=float, default=1.0,
                        help='alpha for mixup/cutmix beta distribution')
    
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug: train 1 epoch')
    # wandb
    parser.add_argument('--wandb', action='store_true', default=True,
                        help='use wandb for visualization')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='your wandb id')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='wandb project name')
    return parser