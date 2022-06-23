from .utils import TwoCropTransform
from .utils import Cutout

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

MEAN = {
    'cifar10': [0.4914, 0.4822, 0.4465],
    'cifar100': [0.5071, 0.4867, 0.4408],
    'imagenet': [0.485, 0.456, 0.406]
}
STD = {
    'cifar10': [0.2023, 0.1994, 0.2010],
    'cifar100':[0.2675, 0.2565, 0.2761],
    'imagenet': [0.229, 0.224, 0.225]
}
SIZE = {
    'cifar10': 32,
    'cifar100': 32,
    'imagenet': 224,
}

NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000
}


def contrastive_loader(args):
    # transformation
    normalize = transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset])
    
    if args.augment_policy == 'sim': # simclr augment
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=SIZE[args.dataset], scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.augment_policy == 'auto': # auto augment
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=SIZE[args.dataset]),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.augment_policy == 'rand':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=SIZE[args.dataset]),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(args.rand_n, args.rand_m),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError(f'Unknown {args.augment_policy}!')

    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.cutout_length))

    # dataset
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='/home/DB/',
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='/home/DB/',
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif args.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder('/home/DB/IMAGENET/train',
                                             TwoCropTransform(train_transform)
                                            )
    else:
        raise ValueError(args.dataset)

    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader, train_sampler


def normal_loader(args):
    normalize = transforms.Normalize(mean=MEAN[args.dataset], std=STD[args.dataset])
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        resize = transforms.RandomCrop(SIZE[args.dataset], padding=4)
    else:
        resize = transforms.RandomResizedCrop(size=SIZE[args.dataset])
    # train dataset
    if args.augment_policy == 'no':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=SIZE[args.dataset], scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.augment_policy == 'sim': # simclr augment
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=SIZE[args.dataset], scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.augment_policy == 'auto':
        train_transform = transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            normalize,
        ])
    elif args.augment_policy == 'rand':
        train_transform = transforms.Compose([
            resize,
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(args.rand_n, args.rand_m),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError()

    if args.erasing:
        train_transform.transforms.append(transforms.RandomErasing(p=args.erasing_p))
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.cutout_length))

    # valid datset
    if args.dataset == 'imagenet':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='/home/DB/',
                                   transform=train_transform,
                                   download=True)
        val_set = datasets.CIFAR10(root='/home/DB/',
                                   transform=val_transform,
                                   train=False,
                                   download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='/home/DB/',
                                   transform=train_transform,
                                   download=True)
        val_set = datasets.CIFAR100(root='/home/DB/',
                                   transform=val_transform,
                                   train=False,
                                   download=True)
    elif args.dataset == 'imagenet':
        train_dataset = datasets.ImageFolder('/home/DB/IMAGENET/train', train_transform)
        val_set = datasets.ImageFolder('/home/DB/IMAGENET/val', val_transform)
    else:
        raise ValueError(args.dataset)
    
    if args.multiprocessing_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler