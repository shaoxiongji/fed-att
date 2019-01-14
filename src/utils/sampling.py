#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6


import numpy as np

def partition(len_dataset, num_users):
    num_items = int(len_dataset/num_users)
    dict_users, all_idxs = {}, [i for i in range(len_dataset)]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

