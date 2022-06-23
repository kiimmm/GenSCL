import numpy as np
import torch
import torch.nn.functional as F


def mix_fn(x, y, alpha, kind):
    if kind == 'mixup':
        return mixup_data(x, y, alpha)
    elif kind == 'cutmix':
        return cutmix_data(x, y, alpha)
    elif kind == 'mixup_cutmix':
        if np.random.rand(1)[0] > 0.5:
            return mixup_data(x, y, alpha)
        else:
            return cutmix_data(x, y, alpha)
    else:
        raise ValueError()


def mix_target(y_a, y_b, lam, num_classes):
    l1 = F.one_hot(y_a, num_classes)
    l2 = F.one_hot(y_b, num_classes)
    return lam * l1 + (1 - lam) * l2


'''
modified from https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
'''
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

'''
modified from https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
'''
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    bsz = x.size()[0]
    index = torch.randperm(bsz, device=x.device)
    
    bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
    mixed_x = x.detach().clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = mixed_x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2