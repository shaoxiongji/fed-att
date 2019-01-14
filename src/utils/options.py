#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import os
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # learning arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--nusers', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate client')
    parser.add_argument('--lr_server', type=float, default=0.001, help='learning rate of server')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--agg', type=str, default='att', help='averaging strategy')
    parser.add_argument('--epsilon', type=float, default=1, help='stepsize')
    parser.add_argument('--ord', type=int, default=2, help='similarity metric')
    parser.add_argument('--dp', type=float, default=0.001, help='differential privacy')
    # model arguments
    parser.add_argument('--d_embed', type=int, default=300, help='embedding dimension')
    parser.add_argument('--d_dict', type=int, default=10000, help='size of the dictionary of embeddings')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    parser.add_argument('--tied', action='store_true', help="Use tied input/output embedding weights: 1 for true")
    parser.add_argument('--rnn_type', type=str, default='GRU', help='type of RNN')
    parser.add_argument("--rnn_hidden", type=int, default=300, help="RNN hidden unit dimensionality")
    parser.add_argument("--rnn_layers", type=int, default=1, help="Number of RNN layers")
    parser.add_argument("--rnn_dropout", type=float, default=0, help="The rate of dropout in RNN layers")

    # other arguments
    parser.add_argument('--dataset', type=str, default='wikitext-2', help='dataset name')
    parser.add_argument('--iid', type=int, default=1, help='whether i.i.d or not, 1 for iid, 0 for non-iid')
    parser.add_argument('--split', type=int, default=0, help='do test split')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose print, 1 for True, 0 for False')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    parser.add_argument('--word_vectors', type=str, default='glove.6B.100d')
    parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args
