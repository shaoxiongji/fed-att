#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplitLM(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdateLM(object):
    def __init__(self, args, dataset, idxs, nround, user):
        self.args = args
        self.round = nround
        self.user = user
        self.loss_func = nn.NLLLoss()
        self.data_loader = DataLoader(DatasetSplitLM(dataset, list(idxs)), batch_size=self.args.local_bs, shuffle=True)

    def update_weights(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        list_loss, list_pp = [], []
        for iter in range(self.args.local_ep):
            for batch_ind, sents in enumerate(self.data_loader):
                net.zero_grad()
                sents.sort(key=lambda l: len(l), reverse=True)
                x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
                y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
                if self.args.gpu != -1:
                    net = net.cuda()
                    x, y = x.cuda(), y.cuda()
                out = net(x)
                loss = self.loss_func(out, y.data)
                loss.backward()
                optimizer.step()
                # Calculate perplexity.
                prob = out.exp()[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
                perplexity = 2 ** prob.log2().neg().mean().item()
                list_loss.append(loss.item())
                list_pp.append(perplexity)
        return {'params': net.cpu().state_dict(),
                'loss': sum(list_loss) / len(list_loss),
                'pp': perplexity}

    def evaluate(self, data_loader, model):
        """ Perplexity of the given data with the given model. """
        model.eval()
        with torch.no_grad():
            entropy_sum = 0
            word_count = 0
            for val_idx, sents in enumerate(data_loader):
                x = nn.utils.rnn.pack_sequence([s[:-1] for s in sents])
                y = nn.utils.rnn.pack_sequence([s[1:] for s in sents])
                if self.args.gpu != -1:
                    x, y = x.cuda(), y.cuda()
                    model = model.cuda()
                out = model(x)
                prob = out.exp()[torch.arange(0, y.data.shape[0], dtype=torch.int64), y.data]
                entropy_sum += prob.log2().neg().sum().item()
                word_count += y.data.shape[0]
        return 2 ** (entropy_sum / word_count)
