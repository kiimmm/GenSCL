import math
import datetime
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchlars import LARS
import wandb

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def set_optimizer(model, args):
    if args.optim_kind == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim_kind == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim_kind == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optim_kind == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    if args.LARS:
        optimizer = LARS(optimizer)
    return optimizer


def accuracy(output, target):
    with torch.no_grad():
        bsz = target.shape[0]
        pred = torch.argmax(output, dim=1)
        acc = 100 * (pred == target).sum() / bsz
    return acc.item()
    
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        warmup_to = eta_min + (args.learning_rate - eta_min) * (
            1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2

        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
            
def save_model(model, optimizer, args, epoch, save_file):
    print(f'==> Saving {save_file}...')
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if not save_file.parent.exists():
        save_file.parent.mkdir()
    torch.save(state, save_file)
    del state

def seed(seed=1):
    """
    Seed for PyTorch reproducibility.
    Arguments:
        seed (int): Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def init_wandb(args):
    wandb.init(
        entity=args.wandb_entity, 
        project=args.wandb_project,
        name=args.desc,
        config=args,
    )
    wandb.run.save()
    return wandb.config

    
def format_time(elapsed):
    """
    Format time for displaying.
    Arguments:
        elapsed: time interval in seconds.
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)