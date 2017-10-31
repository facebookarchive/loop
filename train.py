# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import visdom
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from data import NpzFolder, NpzLoader, TBPTTIter
from model import Loop, MaskedMSE
from utils import create_output_dir, wrap, check_grad


parser = argparse.ArgumentParser(description='PyTorch Loop')
# Env options:
parser.add_argument('--epochs', type=int, default=92, metavar='N',
                    help='number of epochs to train (default: 92)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--expName', type=str, default='vctk', metavar='E',
                    help='Experiment name')
parser.add_argument('--data', default='data/vctk',
                    metavar='D', type=str, help='Data path')
parser.add_argument('--checkpoint', default='',
                    metavar='C', type=str, help='Checkpoint path')
parser.add_argument('--gpu', default=0,
                    metavar='G', type=int, help='GPU device ID')
parser.add_argument('--visualize', action='store_true',
                    help='Visualize train and validation loss.')
# Data options
parser.add_argument('--seq-len', type=int, default=100,
                    help='Sequence length for tbptt')
parser.add_argument('--max-seq-len', type=int, default=1000,
                    help='Max sequence length for tbptt')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--clip-grad', type=float, default=0.5,
                    help='maximum norm of gradient clipping')
parser.add_argument('--ignore-grad', type=float, default=10000.0,
                    help='ignore grad before clipping')
# Model options
parser.add_argument('--vocabulary-size', type=int, default=44,
                    help='Vocabulary size')
parser.add_argument('--output-size', type=int, default=63,
                    help='Size of decoder output vector')
parser.add_argument('--hidden-size', type=int, default=256,
                    help='Hidden layer size')
parser.add_argument('--K', type=int, default=10,
                    help='No. of attention guassians')
parser.add_argument('--noise', type=int, default=4,
                    help='Noise level to use')
parser.add_argument('--attention-alignment', type=float, default=0.05,
                    help='# of features per letter/phoneme')
parser.add_argument('--nspk', type=int, default=22,
                    help='Number of speakers')
parser.add_argument('--mem-size', type=int, default=20,
                    help='Memory number of segments')


# init
args = parser.parse_args()
args.expName = os.path.join('checkpoints', args.expName)
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logging = create_output_dir(args)
vis = visdom.Visdom(env=args.expName)


# data
logging.info("Building dataset.")
train_dataset = NpzFolder(args.data + '/numpy_features', args.nspk == 1)
train_loader = NpzLoader(train_dataset,
                         max_seq_len=args.max_seq_len,
                         batch_size=args.batch_size,
                         num_workers=4,
                         pin_memory=True,
                         shuffle=True)

valid_dataset = NpzFolder(args.data + '/numpy_features_valid', args.nspk == 1)
valid_loader = NpzLoader(valid_dataset,
                         max_seq_len=args.max_seq_len,
                         batch_size=args.batch_size,
                         num_workers=4,
                         pin_memory=True)

logging.info("Dataset ready!")


def train(model, criterion, optimizer, epoch, train_losses):
    total = 0   # Reset every plot_every
    model.train()
    train_enum = tqdm(train_loader, desc='Train epoch %d' % epoch)

    for full_txt, full_feat, spkr in train_enum:
        batch_iter = TBPTTIter(full_txt, full_feat, spkr, args.seq_len)
        batch_total = 0

        for txt, feat, spkr, start in batch_iter:
            input = wrap(txt)
            target = wrap(feat)
            spkr = wrap(spkr)

            # Zero gradients
            if start:
                optimizer.zero_grad()

            # Forward
            output, _ = model([input, spkr], target[0], start)
            loss = criterion(output, target[0], target[1])

            # Backward
            loss.backward()
            if check_grad(model.parameters(), args.clip_grad, args.ignore_grad):
                logging.info('Not a finite gradient or too big, ignoring.')
                optimizer.zero_grad()
                continue
            optimizer.step()

            # Keep track of loss
            batch_total += loss.data[0]

        batch_total = batch_total/len(batch_iter)
        total += batch_total
        train_enum.set_description('Train (loss %.2f) epoch %d' %
                                   (batch_total, epoch))

    avg = total / len(train_loader)
    train_losses.append(avg)
    if args.visualize:
        vis.line(Y=np.asarray(train_losses),
                 X=torch.arange(1, 1 + len(train_losses)),
                 opts=dict(title="Train"),
                 win='Train loss ' + args.expName)

    logging.info('====> Train set loss: {:.4f}'.format(avg))


def evaluate(model, criterion, epoch, eval_losses):
    total = 0
    valid_enum = tqdm(valid_loader, desc='Valid epoch %d' % epoch)

    for txt, feat, spkr in valid_enum:
        input = wrap(txt, volatile=True)
        target = wrap(feat, volatile=True)
        spkr = wrap(spkr, volatile=True)

        output, _ = model([input, spkr], target[0])
        loss = criterion(output, target[0], target[1])

        total += loss.data[0]

        valid_enum.set_description('Valid (loss %.2f) epoch %d' %
                                   (loss.data[0], epoch))

    avg = total / len(valid_loader)
    eval_losses.append(avg)
    if args.visualize:
        vis.line(Y=np.asarray(eval_losses),
                 X=torch.arange(1, 1 + len(eval_losses)),
                 opts=dict(title="Eval"),
                 win='Eval loss ' + args.expName)

    logging.info('====> Test set loss: {:.4f}'.format(avg))
    return avg


def main():
    start_epoch = 1
    model = Loop(args)
    model.cuda()

    if args.checkpoint != '':
        checkpoint_args_path = os.path.dirname(args.checkpoint) + '/args.pth'
        checkpoint_args = torch.load(checkpoint_args_path)

        start_epoch = checkpoint_args[3]
        model.load_state_dict(torch.load(args.checkpoint))

    criterion = MaskedMSE().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Keep track of losses
    train_losses = []
    eval_losses = []
    best_eval = float('inf')

    # Begin!
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(model, criterion, optimizer, epoch, train_losses)
        eval_loss = evaluate(model, criterion, epoch, eval_losses)
        if eval_loss < best_eval:
            torch.save(model.state_dict(), '%s/bestmodel.pth' % (args.expName))
            best_eval = eval_loss

        torch.save(model.state_dict(), '%s/lastmodel.pth' % (args.expName))
        torch.save([args, train_losses, eval_losses, epoch],
                   '%s/args.pth' % (args.expName))


if __name__ == '__main__':
    main()
