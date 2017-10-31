# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from collections import defaultdict
import numpy as np
import os

import torch
import torch.utils.data as data


# Taken from
# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Dataset.py
def batchify(data):
    out, lengths = None, None

    lengths = [x.size(0) for x in data]
    max_length = max(lengths)

    if data[0].dim() == 1:
        out = data[0].new(len(data), max_length).fill_(0)
        for i in range(len(data)):
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data[i])
    else:
        feat_size = data[0].size(1)
        out = data[0].new(len(data), max_length, feat_size).fill_(0)
        for i in range(len(data)):
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data[i])

    return out, lengths


def collate_by_input_length(batch, max_seq_len):
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        return batchify(batch)
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    else:
        new_batch = [x for x in batch if x[1].size(0) < max_seq_len]
        if len(batch) == 0:
            return (None, None), (None, None), None

        batch = new_batch
        transposed = zip(*batch)
        (srcBatch, srcLengths), (tgtBatch, tgtLengths), speakers = \
            [collate_by_input_length(samples, max_seq_len)
                for samples in transposed]

        # within batch sorting by decreasing length for variable length rnns
        batch = zip(srcBatch, tgtBatch, tgtLengths, speakers)
        batch, srcLengths = zip(*sorted(zip(batch, srcLengths),
                                        key=lambda x: -x[1]))
        srcBatch, tgtBatch, tgtLengths, speakers = zip(*batch)

        srcBatch = torch.stack(srcBatch, 0).transpose(0, 1).contiguous()
        tgtBatch = torch.stack(tgtBatch, 0).transpose(0, 1).contiguous()
        srcLengths = torch.LongTensor(srcLengths)
        tgtLengths = torch.LongTensor(tgtLengths)
        speakers = torch.LongTensor(speakers).view(-1, 1)

        return (srcBatch, srcLengths), (tgtBatch, tgtLengths), speakers

    raise TypeError(("batch must contain tensors, numbers, dicts or \
                     lists; found {}".format(type(batch[0]))))


class NpzFolder(data.Dataset):
    NPZ_EXTENSION = 'npz'

    def __init__(self, root, single_spkr=False):
        self.root = root
        self.npzs = self.make_dataset(self.root)

        if len(self.npzs) == 0:
            raise(RuntimeError("Found 0 npz in subfolders of: " + root + "\n"
                               "Supported image extensions are: " +
                               self.NPZ_EXTENSION))

        if single_spkr:
            self.speakers = defaultdict(lambda: 0)
        else:
            self.speakers = []
            for fname in self.npzs:
                self.speakers += [os.path.basename(fname).split('_')[0]]
            self.speakers = list(set(self.speakers))
            self.speakers.sort()
            self.speakers = {v: i for i, v in enumerate(self.speakers)}

        code2phone = np.load(self.npzs[0])['code2phone']
        self.dict = {v: k for k, v in enumerate(code2phone)}

    def __getitem__(self, index):
        path = self.npzs[index]
        txt, feat, spkr = self.loader(path)

        return txt, feat, self.speakers[spkr]

    def __len__(self):
        return len(self.npzs)

    def make_dataset(self, dir):
        images = []

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.NPZ_EXTENSION in fname:
                    path = os.path.join(root, fname)
                    images.append(path)

        return images

    def loader(self, path):
        feat = np.load(path)

        txt = feat['phonemes'].astype('int64')
        txt = torch.from_numpy(txt)

        audio = feat['audio_features']
        audio = torch.from_numpy(audio)

        spkr = os.path.basename(path).split('_')[0]

        return txt, audio, spkr


class NpzLoader(data.DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = partial(collate_by_input_length,
                                       max_seq_len=kwargs['max_seq_len'])
        del kwargs['max_seq_len']

        data.DataLoader.__init__(self, *args, **kwargs)


class TBPTTIter(object):
    """
    Iterator for truncated batch propagation through time(tbptt) training.
    Target sequence is segmented while input sequence remains the same.
    """
    def __init__(self, src, trgt, spkr, seq_len):
        self.seq_len = seq_len
        self.start = True

        self.speakers = spkr
        self.srcBatch = src[0]
        self.srcLenths = src[1]

        # split batch
        self.tgtBatch = list(torch.split(trgt[0], self.seq_len, 0))
        self.tgtBatch.reverse()
        self.len = len(self.tgtBatch)

        # split length list
        batch_seq_len = len(self.tgtBatch)
        self.tgtLenths = [self.split_length(l, batch_seq_len) for l in trgt[1]]
        self.tgtLenths = torch.stack(self.tgtLenths)
        self.tgtLenths = list(torch.split(self.tgtLenths, 1, 1))
        self.tgtLenths = [x.squeeze() for x in self.tgtLenths]
        self.tgtLenths.reverse()

        assert len(self.tgtLenths) == len(self.tgtBatch)

    def split_length(self, seq_size, batch_seq_len):
        seq = [self.seq_len] * (seq_size / self.seq_len)
        if seq_size % self.seq_len != 0:
            seq += [seq_size % self.seq_len]
        seq += [0] * (batch_seq_len - len(seq))
        return torch.LongTensor(seq)

    def __next__(self):
        if len(self.tgtBatch) == 0:
            raise StopIteration()

        if self.len > len(self.tgtBatch):
            self.start = False

        return (self.srcBatch, self.srcLenths), \
               (self.tgtBatch.pop(), self.tgtLenths.pop()), \
               self.speakers, self.start

    next = __next__

    def __iter__(self):
        return self

    def __len__(self):
        return self.len
