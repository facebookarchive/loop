# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import numpy as np
import phonemizer
import string

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
    seperator = phonemizer.separator.Separator('', '', ' ')
    ph = phonemizer.phonemize(text, separator=seperator)
    ph = ph.split(' ')
    ph.remove('')

    result = [char2code[p] for p in ph]
    return torch.LongTensor(result)


def trim_pred(out, attn):
    tq = attn.abs().sum(1).data

    for stopi in range(1, tq.size(0)):
        col_sum = attn[:stopi, :].abs().sum(0).data.squeeze()
        if tq[stopi][0] < 0.5 and col_sum[-1] > 4:
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

    char2code = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 3, 'aw': 4, 'ax': 5,  'ay': 6,
                 'b': 7, 'ch': 8, 'd': 9, 'dh': 10, 'eh': 11, 'er': 12, 'ey': 13,
                 'f': 14, 'g': 15, 'hh': 16, 'i': 17, 'ih': 18, 'iy': 19, 'jh': 20,
                 'k': 21, 'l': 22, 'm': 23, 'n': 24, 'ng': 25, 'ow': 26, 'oy': 27,
                 'p': 28, 'pau': 29, 'r': 30, 's': 31, 'sh': 32, 'ssil': 33,
                 't': 34, 'th': 35, 'uh': 36, 'uw': 37, 'v': 38, 'w': 39, 'y': 40,
                 'z': 41}
    nspkr = train_args.nspk

    norm_path = None
    if os.path.exists(train_args.data + '/norm_info/norm.dat'):
        norm_path = train_args.data + '/norm_info/norm.dat'
    elif os.path.exists(os.path.dirname(args.checkpoint) + '/norm.dat'):
        norm_path = os.path.dirname(args.checkpoint) + '/norm.dat'
    else:
        print('ERROR: Failed to find norm file.')
        return
    train_args.noise = 0

    model = Loop(train_args)
    model.load_state_dict(weights)
    if args.gpu >= 0:
        model.cuda()
    model.eval()

    if args.spkr not in range(nspkr):
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
        feat = torch.FloatTensor(txt.size(0)*20, 63)
        spkr = torch.LongTensor([args.spkr])

        txt = Variable(txt.unsqueeze(1), volatile=True)
        feat = Variable(feat.unsqueeze(1), volatile=True)
        spkr = Variable(spkr, volatile=True)

        # slugify input string to file name
        fname = args.text.replace(' ', '_')
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        fname = ''.join(c for c in fname if c in valid_chars)

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
