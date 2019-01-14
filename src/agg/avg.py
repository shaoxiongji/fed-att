#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import torch


def average_weights(w, dp):
    """
    Federated averaging
    :param w: list of client model parameters
    :param dp: magnitude of randomization
    :return: updated server model parameters
    """
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k] + w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w)) + torch.mul(torch.randn(w_avg[k].shape), dp)
    return w_avg


