#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6


import os
from torch.utils.data import Dataset


class Vocab(object):
    """ Converts word tokens to indices, and vice versa. """
    def __init__(self):
        super().__init__()
        self.stoi = {}
        self.itos = []
        self._counts = []

    def count(self, word):
        """
        Returns the count of a word occurs in the dict.
        :param word: a word
        :return: the count of the given word in the dict.
        """
        return self._counts[self[word]]

    def add(self, word):
        """
        Adds the given word to the dict, and returns it's index.
        If the given word is already in the dict, increases it's count.
        :param word: a word
        :return: the index of the given word
        """
        ind = self.stoi.get(word, None)  # get index, if doesn't exit, return None
        if ind is None:
            ind = len(self.itos)
            self.itos .append(word)
            self.stoi[word] = ind
            self._counts.append(1)
        else:
            self._counts[ind] += 1
        return ind

    def word_counts(self):
        """ Returns a list of tuples (word, count) sorted descending. """
        return sorted(zip(self.itos, self._counts), key=lambda t: t[1], reverse=True)

    def __len__(self):
        return len(self.stoi)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.itos[key]
        else:
            return self.stoi[key]

    def __setitem__(self, word, value):
        raise Exception("Can't set items directly, use add(word) instead")

    def __delitem__(self, word):
        raise Exception("Can't delete items from index.")

    def __contains__(self, word):
        if not isinstance(word, str):
            raise Exception("Presence checks only allowed with words")
        return word in self.stoi


class DatasetLM(Dataset):
    def __init__(self, dataset, train_val_test, index):
        """ Loads the data at the given path using the given index (maps tokens to indices).
        Returns a list of sentences where each is a list of token indices.
        """
        assert train_val_test in ('train', 'test', 'valid')
        if 'wiki' in dataset:
            path = os.path.join('../data/{}'.format(dataset), 'wiki.' + train_val_test + '.tokens')
        elif 'ptb' in dataset:
            path = os.path.join('../data/{}'.format(dataset), 'ptb.' + train_val_test + '.txt')
        elif 'reddit' in dataset:
            path = os.path.join('../data/{}'.format(dataset), 'reddit.'+train_val_test+'.txt')
        start = index.add('<SOS>')
        self.list_sent = []
        with open(path, "r") as f:
            for paragraph in f:
                for sentence in paragraph.split(" . "):
                    tokens = sentence.split()
                    if not tokens:
                        continue
                    sentence = [index.add('<SOS>')]
                    sentence.extend(index.add(t.lower()) for t in tokens)
                    sentence.append(index.add('EOS'))
                    self.list_sent.append(sentence)

    def __len__(self):
        return len(self.list_sent)

    def __getitem__(self, idx):
        return self.list_sent[idx]
