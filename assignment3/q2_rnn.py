#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2: Recurrent neural nets for NER
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import logging
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from ner_model import NERModel
from defs import LBLS
from q2_rnn_cell import RNNCell
from q3_gru_cell import GRUCell
logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_word_features = 2 # Number of features for every word in the input.
    window_size = 1
    n_features = (2 * window_size + 1) * n_word_features # Number of features for every word in the input.
    max_length = 120 # longest sequence to parse
    n_classes = 5
    dropout = 0.5
    embed_size = 50
    hidden_size = 300
    batch_size = 32
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001

    def __init__(self, args):
        self.cell = args.cell

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"

def pad_sequences(data, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    TODO: In the code below, for every sentence, labels pair in @data,
    (a) create a new sentence which appends zero feature vectors until
    the sentence is of length @max_length. If the sentence is longer
    than @max_length, simply truncate the sentence to be @max_length
    long.
    (b) create a new label sequence similarly.
    (c) create a _masking_ sequence that has a True wherever there was a
    token in the original sequence, and a False for every padded input.

    Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
    0, 0], and max_length = 5, we would construct
        - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
        - a new label seqeunce: [1, 0, 0, 4, 4], and
        - a masking seqeunce: [True, True, True, False, False].

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels. 
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sentence', labels', mask).
        Each of sentence', labels' and mask are of length @max_length.
        See the example above for more details.
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_features
    zero_label = 4 # corresponds to the 'O' tag

    for sentence, labels in data:
        sen_len = len(sentence)
        new_sentence = sentence[:max_length] + [zero_vector] * (max_length - sen_len)
        new_label = labels[:max_length] + [zero_label] * (max_length - sen_len)
        mask = ([True] * sen_len)[:max_length] + [False] * (max_length - sen_len)
        ret.append((new_sentence, new_label, mask))
    return ret


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)


class RNNModel(NERModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(RNNModel, self).__init__(helper, config, report)
        self.max_length = min(Config.max_length, helper.max_length)
        Config.max_length = self.max_length # Just in case people make a mistake.

        self.pretrained_embeddings = nn.Embedding(pretrained_embeddings.shape[0],
                                                  pretrained_embeddings.shape[1]).float()
        self.pretrained_embeddings.weight = nn.Parameter(torch.from_numpy(pretrained_embeddings).float())

        if self.config.cell == "rnn":
            self.cell = RNNCell(Config.n_features * Config.embed_size, Config.hidden_size)
        elif self.config.cell == "gru":
            self.cell = GRUCell(Config.n_features * Config.embed_size, Config.hidden_size)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)
        # Defining placeholders.
        self.Ub = nn.Linear(Config.hidden_size, Config.n_classes).float()

    def forward(self, input):
        x = self.pretrained_embeddings(input).view((-1, self.max_length, Config.n_features * Config.embed_size))
        preds = []
        h = torch.zeros(1, Config.hidden_size)

        for time_step in range(self.max_length):
            output, new_state = self.cell(x[:,time_step,:], h)
            out_drop = F.dropout(output, training=self.training)
            pred = self.Ub(out_drop)
            preds.append(pred)
        preds = torch.stack(preds)
        preds = torch.transpose(preds, 0, 1)
        assert list(preds.size())[1:] == [self.max_length, self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, self.config.n_classes], list(preds.size()))
        return preds

    def take_loss(self, labels, preds, mask):
        y_onehot = torch.FloatTensor(*list(preds.size()))

        # In your for loop
        y_onehot.zero_()
        labels = labels.unsqueeze(2)
        y_onehot.scatter_(2, labels, 1)
        mask = mask.unsqueeze(2)
        masked_y = torch.masked_select(y_onehot, mask)
        masked_pred = torch.masked_select(preds, mask)

        loss = torch.sum(-masked_y * F.log_softmax(masked_pred, -1), -1).mean()
        return loss

    def preprocess_sequence_data(self, examples):
        def featurize_windows(data, start, end, window_size = 1):
            """Uses the input sequences in @data to construct new windowed data points.
            """
            ret = []
            for sentence, labels in data:
                from util import window_iterator
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(sum(window, []))
                ret.append((sentence_, labels))
            return ret

        examples = featurize_windows(examples, self.helper.START, self.helper.END)
        return pad_sequences(examples, self.max_length)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, inputs_batch, mask_batch):
        self.eval()
        inputs_batch = torch.from_numpy(inputs_batch)
        return np.argmax(self(inputs_batch).detach().numpy(), axis=2)

    def train_on_batch(self, optimizer, inputs_batch, labels_batch, mask_batch):
        self.train()
        optimizer.zero_grad()
        train_x = torch.from_numpy(inputs_batch)
        train_mask = torch.from_numpy(mask_batch.astype(np.uint8))
        train_y = torch.from_numpy(labels_batch)
        pred = self(train_x)
        loss = self.take_loss(train_y, pred, train_mask)
        loss.backward()
        optimizer.step()
        loss = np.mean(loss.detach().numpy())
        return loss



def test_pad_sequences():
    Config.n_features = 2
    data = [
        ([[4,1], [6,0], [7,0]], [1, 0, 0]),
        ([[3,0], [3,4], [4,5], [5,3], [3,4]], [0, 1, 0, 2, 3]),
        ]
    ret = [
        ([[4,1], [6,0], [7,0], [0,0]], [1, 0, 0, 4], [True, True, True, False]),
        ([[3,0], [3,4], [4,5], [5,3]], [0, 1, 0, 2], [True, True, True, True])
        ]

    ret_ = pad_sequences(data, 4)
    assert len(ret_) == 2, "Did not process all examples: expected {} results, but got {}.".format(2, len(ret_))
    for i in range(2):
        assert len(ret_[i]) == 3, "Did not populate return values corrected: expected {} items, but got {}.".format(3, len(ret_[i]))
        for j in range(3):
            assert ret_[i][j] == ret[i][j], "Expected {}, but got {} for {}-th entry of {}-th example".format(ret[i][j], ret_[i][j], j, i)

def do_test1(_):
    logger.info("Testing pad_sequences")
    test_pad_sequences()
    logger.info("Passed!")

def do_test2(args):
    logger.info("Testing implementation of RNNModel")
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    logger.info("Building model...",)
    start = time.time()
    model = RNNModel(helper, config, embeddings)
    model.apply(init_weights)
    logger.info("took %.2f seconds", time.time() - start)
    model.fit(train, dev)

    logger.info("Model did not crash!")
    logger.info("Passed!")

def do_train(args):
    # Set up some parameters.
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)

    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    logger.info("Building model...",)
    start = time.time()
    model = RNNModel(helper, config, embeddings)
    model.apply(init_weights)
    logger.info("took %.2f seconds", time.time() - start)

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
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_test2)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/train.conll", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/dev.conll", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_train)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
