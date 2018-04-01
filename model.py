# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


def getLinear(dim_in, dim_out):
    return nn.Sequential(nn.Linear(dim_in, dim_in/10),
                         nn.ReLU(),
                         nn.Linear(dim_in/10, dim_out))


class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.criterion = nn.MSELoss(size_average=False)

    # Taken from
    # https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation
    @staticmethod
    def _sequence_mask(sequence_length, max_len):
        batch_size = sequence_length.size(0)
        seq_range = torch.arange(0, max_len).long()
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = Variable(seq_range_expand)
        if sequence_length.is_cuda:
            seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = sequence_length.unsqueeze(1) \
                                           .expand_as(seq_range_expand)
        return (seq_range_expand < seq_length_expand).t().float()

    def forward(self, input, target, lengths):
        max_len = input.size(0)
        mask = self._sequence_mask(lengths, max_len).unsqueeze(2)
        mask_ = mask.expand_as(input)
        self.loss = self.criterion(input*mask_, target*mask_)
        self.loss = self.loss / mask.sum()
        return self.loss


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.hidden_size = opt.hidden_size
        self.vocabulary_size = opt.vocabulary_size
        self.nspk = opt.nspk
        self.lut_p = nn.Embedding(self.vocabulary_size,
                                  self.hidden_size,
                                  max_norm=1.0)
        self.lut_s = nn.Embedding(self.nspk,
                                  self.hidden_size,
                                  max_norm=1.0)

    def forward(self, input, speakers):
        if isinstance(input, tuple):
            lengths = input[1].data.view(-1).tolist()
            outputs = pack(self.lut_p(input[0]), lengths)
        else:
            outputs = self.lut_p(input)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]

        ident = self.lut_s(speakers)
        if ident.dim() == 3:
            ident = ident.squeeze(1)

        return outputs, ident


class GravesAttention(nn.Module):
    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, batch_size, mem_elem, K, attention_alignment):
        super(GravesAttention, self).__init__()
        self.K = K
        self.attention_alignment = attention_alignment
        self.epsilon = 1e-5

        self.sm = nn.Softmax()
        self.N_a = getLinear(mem_elem, 3*K)
        self.J = Variable(torch.arange(0, 500)
                               .expand_as(torch.Tensor(batch_size,
                                          self.K,
                                          500)),
                          requires_grad=False)

    def forward(self, C, context, mu_tm1):
        gbk_t = self.N_a(C.view(C.size(0), C.size(1) * C.size(2)))
        gbk_t = gbk_t.view(gbk_t.size(0), -1, self.K)

        # attention model parameters
        g_t = gbk_t[:, 0, :]
        b_t = gbk_t[:, 1, :]
        k_t = gbk_t[:, 2, :]

        # attention GMM parameters
        g_t = self.sm(g_t) + self.epsilon
        sig_t = torch.exp(b_t) + self.epsilon
        mu_t = mu_tm1 + self.attention_alignment * torch.exp(k_t)

        g_t = g_t.unsqueeze(2).expand(g_t.size(0),
                                      g_t.size(1),
                                      context.size(1))
        sig_t = sig_t.unsqueeze(2).expand_as(g_t)
        mu_t_ = mu_t.unsqueeze(2).expand_as(g_t)
        j = self.J[:g_t.size(0), :, :context.size(1)]

        # attention weights
        phi_t = g_t * torch.exp(-0.5 * sig_t * (mu_t_ - j)**2)
        alpha_t = self.COEF * torch.sum(phi_t, 1)

        c_t = torch.bmm(alpha_t, context).transpose(0, 1).squeeze(0)
        return c_t, mu_t, alpha_t


class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.K = opt.K
        self.hidden_size = opt.hidden_size
        self.output_size = opt.output_size

        self.mem_size = opt.mem_size
        self.mem_feat_size = opt.output_size + opt.hidden_size
        self.mem_elem = self.mem_size * self.mem_feat_size

        self.attn = GravesAttention(opt.batch_size,
                                    self.mem_elem,
                                    self.K,
                                    opt.attention_alignment)

        self.N_o = getLinear(self.mem_elem, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.output_size)
        self.N_u = getLinear(self.mem_elem, self.mem_feat_size)

        self.F_u = nn.Linear(self.hidden_size,  self.hidden_size)
        self.F_o = nn.Linear(self.hidden_size,  self.hidden_size)

    def init_buffer(self, ident, start=True):
        mem_feat_size = self.hidden_size + self.output_size
        batch_size = ident.size(0)

        if start:
            self.mu_t = Variable(ident.data.new(batch_size, self.K).zero_())
            self.S_t = Variable(ident.data.new(batch_size,
                                               mem_feat_size,
                                               self.mem_size).zero_())

            # initialize with identity
            self.S_t[:, :self.hidden_size, :] = ident.unsqueeze(2) \
                                                     .expand(ident.size(0),
                                                             ident.size(1),
                                                             self.mem_size)
        else:
            self.mu_t = self.mu_t.detach()
            self.S_t = self.S_t.detach()

    def update_buffer(self, S_tm1, c_t, o_tm1, ident):
        # concat previous output & context
        idt = torch.tanh(self.F_u(ident))
        o_tm1 = o_tm1.squeeze(0)
        z_t = torch.cat([c_t + idt, o_tm1/30], 1)
        z_t = z_t.unsqueeze(2)
        Sp = torch.cat([z_t, S_tm1[:, :, :-1]], 2)

        # update S
        u = self.N_u(Sp.view(Sp.size(0), -1))
        u[:, :idt.size(1)] = u[:, :idt.size(1)] + idt
        u = u.unsqueeze(2)
        S = torch.cat([u, S_tm1[:, :, :-1]], 2)

        return S

    def forward(self, x, ident, context, start=True):
        out, attns = [], []
        o_t = x[0]
        self.init_buffer(ident, start)

        for o_tm1 in torch.split(x, 1):
            if not self.training:
                o_tm1 = o_t.unsqueeze(0)

            # predict weighted context based on S
            c_t, mu_t, alpha_t = self.attn(self.S_t,
                                           context.transpose(0, 1),
                                           self.mu_t)

            # advance mu and update buffer
            self.S_t = self.update_buffer(self.S_t, c_t, o_tm1, ident)
            self.mu_t = mu_t

            # predict next time step based on buffer content
            ot_out = self.N_o(self.S_t.view(self.S_t.size(0), -1))
            sp_out = self.F_o(ident)
            o_t = self.output(ot_out + sp_out)

            out += [o_t]
            attns += [alpha_t.squeeze()]

        out_seq = torch.stack(out)
        attns_seq = torch.stack(attns)

        return out_seq, attns_seq


class Loop(nn.Module):
    def __init__(self, opt):
        super(Loop, self).__init__()
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)
        self.noise = opt.noise
        self.output_size = opt.output_size

    def init_input(self, tgt, start):
        if start:
            self.x_tm1 = torch.zeros(1, tgt.size(1), tgt.size(2)).type_as(tgt.data)

        if tgt.size(0) > 1:
            inp = torch.cat([self.x_tm1, tgt[:-1].data])
        else:
            inp = self.x_tm1

        if self.noise > 0:
            noise = tgt.data.new(inp.size()).normal_(0, self.noise)
            inp += noise

        if not self.training:
            inp.zero_()

        self.x_tm1 = tgt[-1].data.unsqueeze(0)
        return Variable(inp)

    def cuda(self, device_id=None):
        nn.Module.cuda(self, device_id)
        self.decoder.attn.J = self.decoder.attn.J.cuda(device_id)

    def forward(self, src, tgt, start=True):
        x = self.init_input(tgt, start)

        context, ident = self.encoder(src[0], src[1])
        out, attn = self.decoder(x, ident, context, start)

        return out, attn
