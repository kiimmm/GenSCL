import sys
import time
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
try:
    import wandb
except ImportError:
    pass

from utils import AverageMeter
from utils import adjust_learning_rate, warmup_learning_rate, get_learning_rate
from utils import set_optimizer, save_model
from utils import seed, format_time
from utils import init_wandb

from networks.resnet_big import SupConResNet
from loss import GenSupConLoss
from mix import mix_fn, mix_target
from data import contrastive_loader, NUM_CLASSES
from parser import genscl_parser



# load encoder (student)
def set_model(args):
    model = SupConResNet(name=args.model)
    
    if args.KD: # load teacher model
        teacher = timm.create_model(args.teacher_kind, pretrained=False)
        teacher.reset_classifier(NUM_CLASSES[args.dataset])
        out_ch = teacher.conv_stem.out_channels
        teacher.conv_stem = torch.nn.Conv2d(3, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        teacher_ckpt = torch.load(Path(args.teacher_path)/args.teacher_ckpt, map_location='cpu')
        teacher.load_state_dict(teacher_ckpt['state_dict'])
        teacher.eval()
    else:
        teacher = None

    criterion = GenSupConLoss(temperature=args.temp)
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
            if teacher: teacher = torch.nn.DataParallel(teacher)
        if not args.resume is None: # resume from previous ckpt
            load_fn = Path(args.save_root)/args.desc/f'ckpt_{args.resume}.pth'
            ckpt = torch.load(load_fn, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
            print(f'=> Successfully loading {load_fn}!')
            args.start_epoch = ckpt['epoch'] + 1
        else:
            args.start_epoch = 1
            
        model = model.cuda()
        if teacher: teacher = teacher.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, teacher, criterion


def train(loader, model, teacher, criterion, optimizer, epoch, args):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    
    for idx, (images, targets) in enumerate(loader):
        data_time.update(time.time() - end)
        warmup_learning_rate(args, epoch, idx, len(loader), optimizer)
        
        bsz = targets.shape[0]
        im_q, im_k = images
        if torch.cuda.is_available():
            im_q = im_q.cuda(non_blocking=True)
            im_k = im_k.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            
        if args.mix: # image-based regularizations
            im_q, y0a, y0b, lam0 = mix_fn(im_q, targets, args.mix_alpha, args.mix)
            im_k, y1a, y1b, lam1 = mix_fn(im_k, targets, args.mix_alpha, args.mix)
            images = torch.cat([im_q, im_k], dim=0)
            l_q = mix_target(y0a, y0b, lam0, NUM_CLASSES[args.dataset])
            l_k = mix_target(y1a, y1b, lam1, NUM_CLASSES[args.dataset])
        else:
            images = torch.cat([im_q, im_k], dim=0)
            l_q = F.one_hot(targets, NUM_CLASSES[args.dataset])
            l_k = l_q
        
        if teacher: # KD
            with torch.no_grad():
                with autocast():
                    preds = F.softmax(teacher(images) / args.KD_temp, dim=1)
                    teacher_q, teacher_k = torch.split(preds, [bsz, bsz], dim=0)
                
        # forward
        features = model(images)
        features = torch.split(features, [bsz, bsz], dim=0)

        if teacher:
            if args.KD_alpha == float('inf'): # only learn from teacher's prediction
                loss = criterion(features, [teacher_q, teacher_k])
            else:
                loss = criterion(features, [l_q, l_k]) + args.KD_alpha * criterion(features, [teacher_q, teacher_k])
        else: # no KD
            loss = criterion(features, [l_q, l_k])

        
        losses.update(loss.item(), bsz)
        # backwaqrd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()
            
    res = {
        'trn_loss': losses.avg,
        'learning_rate': get_learning_rate(optimizer)
    }
    return res


def default_desc(args):
    if args.desc is None:
        desc = args.dataset + '_'
        desc += f'bsz_{args.batch_size}_'
        if args.cutout:
            desc += 'cutout_'
        if args.mix:
            desc += f'{args.mix}_{args.mix_alpha}_'
        if args.KD:
            desc += f'KD_{args.KD_alpha}_'
        desc += f'{args.optim_kind}_lr_{args.learning_rate}'
        args.desc = desc
    return args


def main():
    parser = genscl_parser()
    args = parser.parse_args()
    args = default_desc(args)
    seed(args.seed)

    if args.debug:
        args.epochs = 1
    elif args.wandb:
        init_wandb(args)
    save_dir = Path(args.save_root)/args.desc
    
    # build data loader
    train_loader, _ = contrastive_loader(args)

    # build model and criterion
    model, teacher, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(model, args)

    # train
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        res = train(train_loader, model, teacher, criterion, optimizer, epoch, args)
        time2 = time.time()
        print(f'epoch {epoch}, total time {format_time(time2 - time1)}')
        if not args.debug and args.wandb:
            wandb.log(res, step=epoch)
        
        if (epoch % args.save_freq == 0) and not args.debug:
            save_fn = save_dir/f'ckpt_{epoch}.pth'
            save_model(model, optimizer, args, epoch, save_fn)

    if not args.debug:
        save_fn = save_dir/f'ckpt_last.pth'
        save_model(model, optimizer, args, args.epochs, save_fn)
        if args.wandb:
            wandb.finish()

if __name__ == '__main__':
    main()