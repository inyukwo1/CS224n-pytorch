#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3(d): Grooving with GRUs
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

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class GRUCell(torch.nn.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, input_size, state_size):
        super(GRUCell, self).__init__(hidden_size=state_size, input_size=input_size)
        self.input_size = input_size
        self._state_size = state_size
        self.Wz = nn.Linear(input_size, state_size).float()
        self.Uz = nn.Linear(state_size, state_size, bias=False).float()
        self.Wr = nn.Linear(input_size, state_size).float()
        self.Ur = nn.Linear(state_size, state_size, bias=False).float()
        self.Wo = nn.Linear(input_size, state_size).float()
        self.Uo = nn.Linear(state_size, state_size, bias=False).float()

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._state_size

    def forward(self, *inputs):
        input, state = inputs
        z_t = F.sigmoid(self.Wz(input) + self.Uz(state))
        r_t = F.sigmoid(self.Wr(input) + self.Ur(state))
        o_t = F.tanh(self.Wo(input) + r_t * self.Uo(state))
        new_state = z_t * state + (1 - z_t) * o_t

        output = new_state
        return output, new_state


def test_initialize(grucell):
    grucell.Wr.weight.data = torch.from_numpy(np.eye(3,2, dtype=np.float32)).t()
    grucell.Ur.weight.data = torch.from_numpy(np.eye(2,2, dtype=np.float32))
    grucell.Wr.bias.data = torch.from_numpy(np.ones(2, dtype=np.float32))
    grucell.Wz.weight.data = torch.from_numpy(np.eye(3,2, dtype=np.float32)).t()
    grucell.Uz.weight.data = torch.from_numpy(np.eye(2,2, dtype=np.float32))
    grucell.Wz.bias.data = torch.from_numpy(np.ones(2, dtype=np.float32))
    grucell.Wo.weight.data = torch.from_numpy(np.eye(3,2, dtype=np.float32)).t()
    grucell.Uo.weight.data = torch.from_numpy(np.eye(2,2, dtype=np.float32))
    grucell.Wo.bias.data = torch.from_numpy(np.ones(2, dtype=np.float32))


def test_gru_cell():
    cell = GRUCell(3, 2)
    test_initialize(cell)
    x = np.array([
        [0.4, 0.5, 0.6],
        [0.3, -0.2, -0.1]], dtype=np.float32)
    h = np.array([
        [0.2, 0.5],
        [-0.3, -0.3]], dtype=np.float32)
    y = np.array([
        [0.320, 0.555],
        [-0.006, 0.020]], dtype=np.float32)
    ht = y
    y_, ht_ = cell(torch.from_numpy(x), torch.from_numpy(h))
    y_ = y_.detach().numpy()
    ht_ = ht_.detach().numpy()

    print("y_ = " + str(y_))
    print("ht_ = " + str(ht_))

    assert np.allclose(y_, ht_), "output and state should be equal."
    assert np.allclose(ht, ht_, atol=1e-2), "new state vector does not seem to be correct."

def do_test(_):
    logger.info("Testing gru_cell")
    test_gru_cell()
    logger.info("Passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests the GRU cell implemented as part of Q3 of Homework 3')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test', help='')
    command_parser.set_defaults(func=do_test)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
