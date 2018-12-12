#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2(c): Recurrent neural nets for NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger("hw3.q2.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.)

class RNNCell(torch.nn.RNNCell):
    """Wrapper around our RNN cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        super(RNNCell, self).__init__(hidden_size=state_size, input_size=input_size)
        self.input_size = input_size
        self._state_size = state_size
        self.Wx = nn.Linear(input_size, state_size)
        self.Wh = nn.Linear(state_size, state_size, bias=False)


    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def forward(self, *inputs):
        """Updates the state using the previous @state and @inputs.
        Remember the RNN equations are:

        h_t = sigmoid(x_t W_x + h_{t-1} W_h + b)

        TODO: In the code below, implement an RNN cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_x, W_h, b to be variables of the apporiate shape
              using the `tf.get_variable' functions. Make sure you use
              the names "W_x", "W_h" and "b"!
            - Compute @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """

        input, state = inputs
        x = self.Wx(input)
        x = x + self.Wh(state)
        new_state = F.sigmoid(x)

        output = new_state
        return output, new_state


def test_initialize(rnncell):
    rnncell.Wx.weight.data = torch.from_numpy(np.eye(3,2, dtype=np.float32)).t()
    rnncell.Wh.weight.data = torch.from_numpy(np.eye(2,2, dtype=np.float32))
    rnncell.Wx.bias.data = torch.from_numpy(np.ones(2, dtype=np.float32))


def test_rnn_cell():
    cell = RNNCell(3, 2)
    test_initialize(cell)
    x = np.array([
        [0.4, 0.5, 0.6],
        [0.3, -0.2, -0.1]], dtype=np.float32)
    h = np.array([
        [0.2, 0.5],
        [-0.3, -0.3]], dtype=np.float32)
    y = np.array([
        [0.832, 0.881],
        [0.731, 0.622]], dtype=np.float32)
    ht = y
    y_, ht_ = cell(torch.from_numpy(x), torch.from_numpy(h))
    y_ = y_.detach().numpy()
    ht_ = ht_.detach().numpy()
    print("y_ = " + str(y_))
    print("ht_ = " + str(ht_))

    assert np.allclose(y_, ht_), "output and state should be equal."
    assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."


def do_test(_):
    logger.info("Testing rnn_cell")
    test_rnn_cell()
    logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the RNN cell implemented as part of Q2 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
