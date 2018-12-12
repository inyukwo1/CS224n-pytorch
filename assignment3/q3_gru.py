#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3: Grooving with GRUs
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import torch
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import Progbar, minibatches
from model import Model

from q3_gru_cell import GRUCell
from q2_rnn_cell import RNNCell

matplotlib.use('TkAgg')
logger = logging.getLogger("hw3.q3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.
    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. Use self.config.? instead of Config.?
    """
    max_length = 20 # Length of sequence used.
    batch_size = 100
    n_epochs = 40
    lr = 0.2
    max_grad_norm = 5.

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)

class SequencePredictor(Model):
    def __init__(self, config):
        super(SequencePredictor, self).__init__()
        self.config = config
        self.grad_norm = None
        # Pick out the cell to use here.
        if self.config.cell == "rnn":
            self.cell = RNNCell(1, 1)
        elif self.config.cell == "gru":
            self.cell = GRUCell(1, 1)
        elif self.config.cell == "lstm":
            pass
        else:
            raise ValueError("Unsupported cell type.")

    def forward(self, input):
        """Runs an rnn on the input using TensorFlows's
        @tf.nn.dynamic_rnn function, and returns the final state as a prediction.

        TODO:
            - Call tf.nn.dynamic_rnn using @cell below. See:
              https://www.tensorflow.org/api_docs/python/nn/recurrent_neural_networks
            - Apply a sigmoid transformation on the final state to
              normalize the inputs between 0 and 1.

        Returns:
            preds: tf.Tensor of shape (batch_size, 1)
        """

        h = torch.zeros(list(input.size())[0], self.cell.state_size).float()
        for time_step in range(list(input.size())[1]):
            output, h = self.cell(input.float()[:,time_step,:], h)
        h = F.sigmoid(h)
        return h #state # preds

    def take_loss(self, labels, preds):
        loss = nn.MSELoss(reduction='sum')
        loss = loss(labels.double(), preds.double()) / 2
        return loss

    def run_epoch(self, optimizer, train):
        prog = Progbar(target=1 + int(len(train) / self.config.batch_size))
        losses, grad_norms = [], []
        for i, [x, y] in enumerate(minibatches(train, self.config.batch_size)):
            optimizer.zero_grad()
            train_x = torch.from_numpy(x)
            train_y = torch.from_numpy(y)
            pred = self(train_x)
            loss = self.take_loss(train_y, pred)


            loss.backward()

            if self.config.clip_gradients:
                grad_norm = torch.nn.utils.clip_grad_norm(self.parameters(recurse=True), self.config.max_grad_norm)
            else:
                grad_norm = []
                for param in self.parameters(recurse=True):
                    if param.grad is not None:
                        grad_norm.append(param.grad.norm())
                grad_norm = np.sum(grad_norm)
            optimizer.step()
            loss = loss.detach().numpy()

            losses.append(loss)
            grad_norms.append(grad_norm)
            prog.update(i + 1, [("train loss", loss),("grad norm", grad_norm)])

        return losses, grad_norms

    def fit(self, train):
        losses, grad_norms = [], []
        optimizer = optim.SGD(self.parameters(), lr=self.config.lr)
        print self
        self.train()
        for epoch in range(self.config.n_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            loss, grad_norm = self.run_epoch(optimizer, train)
            losses.append(loss)
            grad_norms.append(grad_norm)

        return losses, grad_norms


def generate_sequence(max_length=20, n_samples=9999):
    """
    Generates a sequence like a [0]*n a
    """
    seqs = []
    for _ in range(int(n_samples/2)):
        seqs.append(([[0.,]] + ([[0.,]] * (max_length-1)), [0.]))
        seqs.append(([[1.,]] + ([[0.,]] * (max_length-1)), [1.]))
    return seqs

def test_generate_sequence():
    max_length = 20
    for seq, y in generate_sequence(20):
        assert len(seq) == max_length
        assert seq[0] == y

def make_dynamics_plot(args, x, h, ht_rnn, ht_gru, params):
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif')

    Ur, Wr, br, Uz, Wz, bz, Uo, Wo, bo = params

    plt.clf()
    plt.title("""Cell dynamics when x={}:
Ur={:.2f}, Wr={:.2f}, br={:.2f}
Uz={:.2f}, Wz={:.2f}, bz={:.2f}
Uo={:.2f}, Wo={:.2f}, bo={:.2f}""".format(x, Ur[0,0], Wr[0,0], br[0], Uz[0,0], Wz[0,0], bz[0], Uo[0,0], Wo[0,0], bo[0]))

    plt.plot(h, ht_rnn, label="rnn")
    plt.plot(h, ht_gru, label="gru")
    plt.plot(h, h, color='gray', linestyle='--')
    plt.ylabel("$h_{t}$")
    plt.xlabel("$h_{t-1}$")
    plt.legend()
    output_path = "{}-{}-{}.png".format(args.output_prefix, x, "dynamics")
    plt.savefig(output_path)

def initialize(params, cell):
    Ur, Wr, br, Uz, Wz, bz, Uo, Wo, bo = params
    if cell.Wr is not None:
        cell.Wr.weight.data = torch.from_numpy(Wr)
    if cell.Ur is not None:
        cell.Ur.weight.data = torch.from_numpy(Ur)
    if cell.Wr.bias is not None:
        cell.Wr.bias.data = torch.from_numpy(br)

    if cell.Wz is not None:
        cell.Wz.weight.data = torch.from_numpy(Wz)
    if cell.Uz is not None:
        cell.Uz.weight.data = torch.from_numpy(Uz)
    if cell.Wz.bias is not None:
        cell.Wz.bias.data = torch.from_numpy(bz)

    if cell.Wo is not None:
        cell.Wo.weight.data = torch.from_numpy(Wo)
    if cell.Uo is not None:
        cell.Uo.weight.data = torch.from_numpy(Uo)
    if cell.Wo.bias is not None:
        cell.Wo.bias.data = torch.from_numpy(bo)


def compute_cell_dynamics(args):
    def mat(x):
        return np.atleast_2d(np.array(x, dtype=np.float32))
    def vec(x):
        return np.atleast_1d(np.array(x, dtype=np.float32))

    Ur, Wr, Uz, Wz, Uo, Wo = [mat(3 * x) for x in np.random.randn(6)]
    br, bz, bo = [vec(x) for x in np.random.randn(3)]
    params = [Ur, Wr, br, Uz, Wz, bz, Uo, Wo, bo]

    np.random.seed(41)
    gru = GRUCell(1, 1)
    initialize(params, gru)
    rnn = RNNCell(1, 1)
    initialize(params, rnn)

    x = mat(np.zeros(1000)).T
    h = mat(np.linspace(-3, 3, 1000)).T

    y_gru, h_gru = gru(x, h)
    y_rnn, h_rnn = rnn(x, h)
    ht_gru = np.array(h_gru)[0]
    ht_rnn = np.array(h_rnn)[0]
    make_dynamics_plot(args, 0, h, ht_rnn, ht_gru, params)

    x = mat(np.ones(1000)).T
    h = mat(np.linspace(-3, 3, 1000)).T

    y_gru, h_gru = gru(x, h)
    y_rnn, h_rnn = rnn(x, h)
    ht_gru = np.array(h_gru)[0]
    ht_rnn = np.array(h_rnn)[0]
    make_dynamics_plot(args, 1, h, ht_rnn, ht_gru, params)

def make_prediction_plot(args, losses, grad_norms):
    plt.subplot(2, 1, 1)
    plt.title("{} on sequences of length {} ({} gradient clipping)".format(args.cell, args.max_length, "with" if args.clip_gradients else "without"))
    plt.plot(np.arange(losses.size), losses.flatten(), label="Loss")
    plt.ylabel("Loss")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(grad_norms.size), grad_norms.flatten(), label="Gradients")
    plt.ylabel("Gradients")
    plt.xlabel("Minibatch")
    output_path = "{}-{}clip-{}.png".format(args.output_prefix, "" if args.clip_gradients else "no", args.cell)
    plt.savefig(output_path)

def do_sequence_prediction(args):
    # Set up some parameters.
    config = Config()
    config.cell = args.cell
    config.clip_gradients = args.clip_gradients

    # You can change this around, but make sure to reset it to 41 when
    # submitting.
    np.random.seed(21)
    torch.random.manual_seed(109)
    data = generate_sequence(args.max_length)

    logger.info("Building model...",)
    start = time.time()
    model = SequencePredictor(config)
    model.apply(init_weights)
    logger.info("took %.2f seconds", time.time() - start)

    losses, grad_norms = model.fit(data)

    # Plotting code.
    losses, grad_norms = np.array(losses), np.array(grad_norms)
    make_prediction_plot(args, losses, grad_norms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs a sequence model to test latching behavior of memory, e.g. 100000000 -> 1')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('predict', help='Plot prediction behavior of different cells')
    command_parser.add_argument('-c', '--cell', choices=['rnn', 'gru'], default='rnn', help="Type of cell to use")
    command_parser.add_argument('-g', '--clip_gradients', action='store_true', default=False, help="If true, clip gradients")
    command_parser.add_argument('-l', '--max-length', type=int, default=20, help="Length of sequences to generate")
    command_parser.add_argument('-o', '--output-prefix', type=str, default="q3", help="Length of sequences to generate")
    command_parser.set_defaults(func=do_sequence_prediction)

    # Easter egg! Run this function to plot how an RNN or GRU map an
    # input state to an output state.
    command_parser = subparsers.add_parser('dynamics', help="Plot cell's dynamics")
    command_parser.add_argument('-o', '--output-prefix', type=str, default="q3", help="Length of sequences to generate")
    command_parser.set_defaults(func=compute_cell_dynamics)


    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
