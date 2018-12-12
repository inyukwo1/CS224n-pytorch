#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1: A window into NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import sys
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import print_sentence, write_conll
from data_util import load_and_preprocess_data, load_embeddings, read_conll, ModelHelper
from ner_model import NERModel
from defs import LBLS
#from report import Report

logger = logging.getLogger("hw3.q1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

    TODO: Fill in what n_window_features should be, using n_word_features and window_size.
    """
    n_word_features = 2 # Number of features for every word in the input.
    window_size = 1 # The size of the window to use.
    ### YOUR CODE HERE
    n_window_features = n_word_features * (2 * window_size + 1)
    ### END YOUR CODE
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001

    def __init__(self, output_path=None):
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/window/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
        self.conll_output = self.output_path + "window_predictions.conll"


def make_windowed_data(data, start, end, window_size = 1):
    """Uses the input sequences in @data to construct new windowed data points.

    TODO: In the code below, construct a window from each word in the
    input sentence by concatenating the words @window_size to the left
    and @window_size to the right to the word. Finally, add this new
    window data point and its label. to windowed_data.

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels.
        start: the featurized `start' token to be used for windows at the very
            beginning of the sentence.
        end: the featurized `end' token to be used for windows at the very
            end of the sentence.
        window_size: the length of the window to construct.
    Returns:
        a new list of data points, corresponding to each window in the
        sentence. Each data point consists of a list of
        @n_window_features features (corresponding to words from the
        window) to be used in the sentence and its NER label.
        If start=[5,8] and end=[6,8], the above example should return
        the list
        [([5, 8, 1, 9, 2, 9], 1),
         ([1, 9, 2, 9, 3, 8], 1),
         ...
         ]
    """

    windowed_data = []
    for sentence, labels in data:
        for idx, word in enumerate(sentence):
            featured_window = []
            for window_idx in range(-window_size, window_size + 1):
                absolute_idx = idx + window_idx
                if absolute_idx < 0:
                    featured_window.extend(start)
                elif absolute_idx >= len(sentence):
                    featured_window.extend(end)
                else:
                    featured_window.extend(sentence[absolute_idx])
            windowed_data.append((featured_window, labels[idx]))

    return windowed_data


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.)


class WindowModel(NERModel):

    """
    Implements a feedforward neural network with an embedding layer and
    single hidden layer.
    This network will predict what label (e.g. PER) should be given to a
    given token (e.g. Manning) by  using a featurized window around the token.
    """

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(WindowModel, self).__init__(helper, config, report)

        self.pretrained_embeddings = nn.Embedding(pretrained_embeddings.shape[0], pretrained_embeddings.shape[1]).float()
        self.pretrained_embeddings.weight = nn.Parameter(torch.from_numpy(pretrained_embeddings).float())
        self.Wb = nn.Linear(Config.n_window_features * Config.embed_size, Config.hidden_size).float()
        self.Ub = nn.Linear(Config.hidden_size, Config.n_classes).float()

    def forward(self, input):
        x = self.pretrained_embeddings(input).view((-1, Config.n_window_features * Config.embed_size))
        x = F.relu(self.Wb(x))
        x = F.dropout(x, training=self.training)
        pred = self.Ub(x)
        return pred

    def take_loss(self, labels, preds):
        batch_size = labels.numpy().shape[0]
        y_onehot = torch.FloatTensor(batch_size, 5)

        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, labels.view(batch_size, 1), 1)

        loss = torch.sum(-y_onehot * F.log_softmax(preds, -1), -1).mean()
        return loss

    def preprocess_sequence_data(self, examples):
        return make_windowed_data(examples, start=self.helper.START, end=self.helper.END, window_size=self.config.window_size)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        ret = []
        #pdb.set_trace()
        i = 0
        for sentence, labels in examples_raw:
            labels_ = preds[i:i+len(sentence)]
            i += len(sentence)
            ret.append([sentence, labels, labels_])
        return ret


def test_make_windowed_data():
    sentences = [[[1,1], [2,0], [3,3]]]
    sentence_labels = [[1, 2, 3]]
    data = zip(sentences, sentence_labels)
    w_data = make_windowed_data(data, start=[5,0], end=[6,0], window_size=1)

    assert len(w_data) == sum(len(sentence) for sentence in sentences)

    assert w_data == [
        ([5,0] + [1,1] + [2,0], 1,),
        ([1,1] + [2,0] + [3,3], 2,),
        ([2,0] + [3,3] + [6,0], 3,),
        ]

def do_test1(_):
    logger.info("Testing make_windowed_data")
    test_make_windowed_data()
    logger.info("Passed!")

def do_test2(args):
    logger.info("Testing implementation of WindowModel")
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    logger.info("Building model...",)
    start = time.time()
    model = WindowModel(helper, config, embeddings)
    logger.info("took %.2f seconds", time.time() - start)
    model.apply(init_weights)
    model.fit(train, dev)

    logger.info("Model did not crash!")
    logger.info("Passed!")

def do_train(args):
    # Set up some parameters.
    config = Config()
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    logger.info("Building model...", )
    start = time.time()
    model = WindowModel(helper, config, embeddings)
    logger.info("took %.2f seconds", time.time() - start)
    model.apply(init_weights)

    model.fit(train, dev)
    if report:
        report.log_output(model.output(dev_raw))
        report.save()
    else:
        # Save predictions in a text file.
        output = model.output(dev_raw)
        sentences, labels, predictions = zip(*output)
        predictions = [[LBLS[l] for l in preds] for preds in predictions]
        output = zip(sentences, labels, predictions)

        with open(model.config.conll_output, 'w') as f:
            write_conll(f, output)
        with open(model.config.eval_output, 'w') as f:
            for sentence, labels, predictions in output:
                print_sentence(f, sentence, labels, predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test1', help='')
    command_parser.set_defaults(func=do_test1)

    command_parser = subparsers.add_parser('test2', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/tiny.conll", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/tiny.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.set_defaults(func=do_test2)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/train.conll", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/dev.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.set_defaults(func=do_train)
    #I didn't implement evaluate and shell mode
    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
