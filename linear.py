import sys
import time
from pathlib import Path
from contextlib import suppress

import torch
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
try:
    import wandb
except ImportError:
    pass

from utils import AverageMeter
from utils import adjust_learning_rate, warmup_learning_rate, accuracy, get_learning_rate
from utils import set_optimizer, save_model
from utils import init_wandb
from utils import format_time
from utils import seed

from networks.resnet_big import SupConResNet, LinearClassifier
from data import normal_loader, NUM_CLASSES
from mix import mix_fn
from parser import linear_parser


# load trained encoder and build a classifier to train
def set_model(args):
    model = SupConResNet(name=args.model)
    classifier = LinearClassifier(name=args.model, num_classes=NUM_CLASSES[args.dataset])

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    load_fn = Path(args.save_root)/args.pretrained/args.pretrained_ckpt
    ckpt = torch.load(load_fn, map_location='cpu')
    state_dict = ckpt['state_dict']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(loader, model, classifier, criterion, optimizer, epoch, amp_autocast, scaler, args):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, targets) in enumerate(loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        bsz = targets.shape[0]

        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(loader), optimizer)

        # compute loss
        with amp_autocast():
            with torch.no_grad():
                features = model.encoder(images)
            output = classifier(features.detach())
            loss = criterion(output, targets)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, targets)
        top1.update(acc1, bsz)
        
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()
    
    res = {
        'trn_loss': losses.avg,
        'trn_top1_acc': top1.avg,
        'learning_rate': get_learning_rate(optimizer)
    }
    return res

def validate(loader, model, classifier, criterion, amp_autocast, args):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, targets) in enumerate(loader):
            images = images.float().cuda()
            targets = targets.cuda()
            bsz = targets.shape[0]

            # forward
            with amp_autocast():
                output = classifier(model.encoder(images))
                loss = criterion(output, targets)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, targets)
            top1.update(acc1, bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    res = {
        'val_loss': losses.avg,
        'val_top1_acc': top1.avg
    }

    return res


def default_desc(args):
    if args.desc is None:
        desc = args.pretrained + '_linear'
        args.desc = desc
    return args


def main():
    best_acc = 0.
    parser = linear_parser()
    args = parser.parse_args()
    args = default_desc(args)
    seed(args.seed)

    if args.debug:
        args.epochs = 1
    elif args.wandb:
        init_wandb(args)
        
    save_dir = Path(args.save_root)/args.desc
    amp_autocast = autocast if args.amp else suppress
    scaler = GradScaler() if args.amp else None
    # build data loader
    train_loader, val_loader, _ = normal_loader(args)

    # build model and criterion
    model, classifier, criterion = set_model(args)

    # build optimizer
    optimizer = set_optimizer(classifier, args)

    # training routine
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        res = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, amp_autocast, scaler, args)
        time2 = time.time()
        print('Train epoch {}, total time {}, accuracy:{:.2f}'.format(
            epoch, format_time(time2 - time1), res['trn_top1_acc']))

        save_model(classifier, optimizer, args, epoch, save_dir/'ckpt_last.pth')
        
        # eval for one epoch
        val_res = validate(val_loader, model, classifier, criterion, amp_autocast, args)
        val_acc = val_res['val_top1_acc']
        res.update(val_res)
        if not(args.debug) and args.wandb:
            wandb.log(res, step=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            save_model(classifier, optimizer, args, epoch, save_dir/'ckpt_best.pth')

    print('best accuracy: {:.2f}'.format(best_acc))
    if not(args.debug) and args.wandb:
        wandb.log({'val_best_acc': best_acc})
        wandb.finish()
    
if __name__ == '__main__':
    main()