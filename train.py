import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict
from torch.utils.data.dataset import Subset
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import config
from utils import *
from archs import archs
import dataloader
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.simplefilter('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default="/home/kotarokuroda/Documents/xray_dataset_covid19/train")
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--height', type=int, default=114)
    parser.add_argument('--width', type=int, default=114)
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', default='ResNet',
                        choices=archs.__all__,
                        help='model architecture')
    parser.add_argument('--metric', default='adacos',
                        choices=['adacos', 'arcface', 'sphereface', 'cosface', 'softmax'])
    parser.add_argument('--num-features', default=512, type=int,
                        help='dimention of embedded features')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float)
    parser.add_argument('--min-lr', default=1e-6, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--cpu', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    acc1s = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.cpu:
            input = input.cpu()
            target = target.long().cpu()
        else:
            input = input.cuda()
            target = target.long().cuda()

        output = model(input, target)
        loss = criterion(output, target)
        acc1, = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), input.size(0))
        acc1s.update(acc1.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    acc1s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if args.cpu:
                input = input.cpu()
                target = target.long().cpu()
            else:
                input = input.cuda()
                target = target.long().cuda()

            output = model(input, target)
            loss = criterion(output, target)

            acc1, = accuracy(output, target, topk=(1,))

            losses.update(loss.item(), input.size(0))
            acc1s.update(acc1.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
    ])

    return log


def main():
    args = parse_args()
    writer = SummaryWriter(log_dir="./logs")

    if args.name is None:
        args.name = 'mnist_%s_%s_%dd' % (args.arch, args.metric, args.num_features)

    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' % args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)

    # criterion = nn.CrossEntropyLoss().cpu()
    classes = config.classes
    dataset = dataloader.MyDataset(args.train_dir, args.height, args.height, classes)
    if args.cpu:
        criterion = nn.CrossEntropyLoss().cpu()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[args.arch](args, len(classes))

    if args.cpu:
        model = model.cpu()
    else:
        model = model.cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                               T_max=args.epochs, eta_min=args.min_lr)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc1', 'val_loss', 'val_acc1'
    ])
    kf = KFold(n_splits=5, shuffle=True, random_state=2020)
    best_loss = float('inf')
    for _fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
        val_dataset = Subset(dataset, val_index)
        val_loader = DataLoader(val_dataset, 1, shuffle=False)
        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' % (epoch + 1, args.epochs))

            # train for one epoch
            train_log = train(args, train_loader, model, criterion, optimizer)
            # evaluate on validation set
            val_log = validate(args, val_loader, model, criterion)
            scheduler.step()
            print('loss %.4f - acc1 %.4f - val_loss %.4f - val_acc %.4f'
                  % (train_log['loss'], train_log['acc1'], val_log['loss'], val_log['acc1']))

            tmp = pd.Series([
                epoch,
                scheduler.get_lr()[0],
                train_log['loss'],
                train_log['acc1'],
                val_log['loss'],
                val_log['acc1'],
            ], index=['epoch', 'lr', 'loss', 'acc1', 'val_loss', 'val_acc1'])

            log = log.append(tmp, ignore_index=True)
            log.to_csv('models/%s/log.csv' % args.name, index=False)
            writer.add_scalar('train_loss', train_log['loss'], _fold * 10 + epoch + 1)
            writer.add_scalar('train_accuracy', train_log['acc1'], _fold * 10 + epoch + 1)
            writer.add_scalar('val_loss', val_log['loss'], _fold * 10 + epoch + 1)
            writer.add_scalar('val_accuracy', val_log['acc1'], _fold * 10 + epoch + 1)
            if val_log['loss'] < best_loss:
                torch.save(model.state_dict(), 'models/%s/model.pth' % args.name)
                best_loss = val_log['loss']
                print("=> saved best model")

    writer.close()


if __name__ == '__main__':
    main()
