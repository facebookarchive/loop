# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import nltk
import argparse
import random
import numpy as np
from string import digits

import torch
from torch.autograd import Variable

from model import Loop
from data import NpzFolder
from utils import generate_merlin_wav


parser = argparse.ArgumentParser(description='PyTorch Phonological Loop \
                                    Generation')
parser.add_argument('--npz', type=str, default='',
                    help='Dataset sample to generate.')
parser.add_argument('--text', default='',
                    type=str, help='Free text to generate.')
parser.add_argument('--spkr', default=0,
                    type=int, help='Speaker id.')
parser.add_argument('--checkpoint', default='checkpoints/vctk/lastmodel.pth',
                    type=str, help='Model used for generation.')
parser.add_argument('--gpu', default=-1,
                    type=int, help='GPU device ID, use -1 for CPU.')


# init
args = parser.parse_args()
if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)


def text2phone(text, char2code):
    cmudict = nltk.corpus.cmudict.dict()

    result = []
    for word in text.split():
        result += random.choice(cmudict[word])
    result = [str(ph.lower()).translate(None, digits) for ph in result]
    result = [char2code[ph] for ph in result]

    return torch.LongTensor(result)


def trim_pred(out, attn):
    tq = attn.abs().sum(1).data
    for stopi in range(tq.size(0) - 1, -1, -1):
        if tq[stopi][0] > 0.5:
            break

    out = out[:stopi, :]
    attn = attn[:stopi, :]

    return out, attn


def npy_loader_phonemes(path):
    feat = np.load(path)

    txt = feat['phonemes'].astype('int64')
    txt = torch.from_numpy(txt)

    audio = feat['audio_features']
    audio = torch.from_numpy(audio)

    return txt, audio


def main():
    weights = torch.load(args.checkpoint,
                         map_location=lambda storage, loc: storage)
    opt = torch.load(os.path.dirname(args.checkpoint) + '/args.pth')
    train_args = opt[0]

    train_dataset = NpzFolder(train_args.data + '/numpy_features')
    char2code = train_dataset.dict
    spkr2code = train_dataset.speakers

    norm_path = train_args.data + '/norm_info/norm.dat'
    train_args.noise = 0

    model = Loop(train_args)
    model.load_state_dict(weights)
    if args.gpu >= 0:
        model.cuda()
    model.eval()

    if args.spkr not in range(len(spkr2code)):
        print('ERROR: Unknown speaker id: %d.' % args.spkr)
        return

    txt, feat, spkr, output_fname = None, None, None, None
    if args.npz is not '':
        txt, feat = npy_loader_phonemes(args.npz)

        txt = Variable(txt.unsqueeze(1), volatile=True)
        feat = Variable(feat.unsqueeze(1), volatile=True)
        spkr = Variable(torch.LongTensor([args.spkr]), volatile=True)

        fname = os.path.basename(args.npz)[:-4]
        output_fname = fname + '.gen_' + str(args.spkr)
    elif args.text is not '':
        txt = text2phone(args.text, char2code)
        feat = torch.FloatTensor(500, 63)
        spkr = torch.LongTensor([args.spkr])

        txt = Variable(txt.unsqueeze(1), volatile=True)
        feat = Variable(feat.unsqueeze(1), volatile=True)
        spkr = Variable(spkr, volatile=True)

        fname = args.text.replace(' ', '_')
        output_fname = fname + '.gen_' + str(args.spkr)
    else:
        print('ERROR: Must supply npz file path or text as source.')
        return

    if args.gpu >= 0:
        txt = txt.cuda()
        feat = feat.cuda()
        spkr = spkr.cuda()


    out, attn = model([txt, spkr], feat)
    out, attn = trim_pred(out, attn)

    output_dir = os.path.join(os.path.dirname(args.checkpoint), 'results')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generate_merlin_wav(out.data.cpu().numpy(),
                        output_dir,
                        output_fname,
                        norm_path)

    if args.npz is not '':
        output_orig_fname = os.path.basename(args.npz)[:-4] + '.orig'
        generate_merlin_wav(feat[:, 0, :].data.cpu().numpy(),
                            output_dir,
                            output_orig_fname,
                            norm_path)


if __name__ == '__main__':
    main()
