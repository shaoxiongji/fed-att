#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import random
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from Update import LocalUpdateLM
from agg.avg import *
from agg.aggregate import *
from Text import Vocab, DatasetLM
from utils.sampling import partition
from utils.options import args_parser
from Models import RnnLm


def get_batches(data, batch_size):
    """ Yields batches of sentences from 'data', ordered on length. """
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        sentences.sort(key=lambda l: len(l), reverse=True)
        yield [torch.LongTensor(s) for s in sentences]


if __name__ == "__main__":
    args = args_parser()

    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vocab = Vocab()
    dataset_train = DatasetLM(args.dataset, 'train', vocab)
    dataset_val = DatasetLM(args.dataset, 'valid', vocab)
    dataset_test = DatasetLM(args.dataset, 'test', vocab)

    loader_train = DataLoader(dataset=dataset_train, batch_size=args.bs, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, batch_size=args.bs, shuffle=True)
    loader_test = DataLoader(dataset=dataset_test, batch_size=args.bs, shuffle=True)

    dict_users = partition(len_dataset=len(dataset_train), num_users=args.nusers)

    config = args
    config.nvocab = len(vocab.stoi)

    net_glob = RnnLm(config)
    if args.gpu != -1:
        net_glob = net_glob.cuda()
    w_glob = net_glob.cpu().state_dict()

    lr = args.lr
    best_val_loss = None
    model_saved = '../log/{}/model_{}_{}_{}.pt'.format(args.dataset, args.epochs, args.agg, args.frac)
    loss_train = []

    try:
        for epoch in range(args.epochs):
            net_glob.train()
            w_locals, loss_locals, pp_locals = [], [], []
            m = max(int(args.frac * args.nusers), 1)
            idxs_users = np.random.choice(range(args.nusers), m, replace=False)
            for idx in idxs_users:
                local = LocalUpdateLM(args=args, dataset=dataset_train, idxs=dict_users[idx], nround=epoch, user=idx)
                net_glob.load_state_dict(w_glob)
                out_dict = local.update_weights(net=copy.deepcopy(net_glob))
                w_locals.append(copy.deepcopy(out_dict['params']))
                loss_locals.append(copy.deepcopy(out_dict['loss']))
                pp_locals.append(copy.deepcopy(out_dict['pp']))

            # update global weights
            if args.agg == 'avg':
                w_glob = average_weights(w_locals, args.dp)
            elif args.agg == 'att':
                w_glob = aggregate_att(w_locals, w_glob, args.epsilon, args.ord, dp=args.dp)
            else:
                exit('Unrecognized aggregation')
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            if args.epochs % 10 == 0:
                print('\nTrain loss:', loss_avg)
            loss_train.append(loss_avg)

            val_loss = local.evaluate(data_loader=loader_val, model=net_glob)
            print("Epoch {}, Validation ppl: {:.1f}".format(epoch, val_loss))

            if not best_val_loss or val_loss < best_val_loss:
                with open(model_saved, 'wb') as f:
                    torch.save(net_glob, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Existing from training early')

    # Load the best saved model.
    with open(model_saved, 'rb') as f:
        model_best = torch.load(f)

    pp_train = local.evaluate(data_loader=loader_train, model=model_best)
    pp_val = local.evaluate(data_loader=loader_val, model=model_best)
    pp_test = local.evaluate(data_loader=loader_test, model=model_best)

    print("Train perplexity: {:.1f}".format(pp_train))
    print("Val perplexity: {:.1f}".format(pp_val))
    print("Test perplexity: {:.1f}".format(pp_test))
