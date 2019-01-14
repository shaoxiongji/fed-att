#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

from torch import nn
import torch.nn.functional as F


class RnnLm(nn.Module):
    def __init__(self, config):
        super(RnnLm, self).__init__()
        self.args = config
        if not config.tied:
            self.embed = nn.Embedding(config.nvocab, config.d_embed)
        self.encoder = nn.GRU(config.d_embed, config.rnn_hidden, config.rnn_layers,
                          dropout=config.rnn_dropout, bias=True, bidirectional=False)
        self.fc1 = nn.Linear(config.rnn_hidden, config.nvocab, bias=True)

    def get_embedded(self, word_indexes):
        if self.args.tied:
            return self.fc1.weight.index_select(0, word_indexes)
        else:
            return self.embed(word_indexes)

    def forward(self, packed_sents):
        embedded_sents = nn.utils.rnn.PackedSequence(self.get_embedded(packed_sents.data), packed_sents.batch_sizes)
        out_packed_sequence, _ = self.encoder(embedded_sents)
        out = self.fc1(out_packed_sequence.data)
        return F.log_softmax(out, dim=1)
