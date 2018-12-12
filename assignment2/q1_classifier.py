import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from utils.general_utils import get_minibatches


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_samples = 1024
    n_features = 100
    n_classes = 5
    batch_size = 64
    n_epochs = 50
    lr = 1e-4

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.)


class SoftmaxModelPyTorch(nn.Module):
    def __init__(self, config):
        super(SoftmaxModelPyTorch, self).__init__()

        self.config = config
        self.Wb = nn.Linear(Config.n_features, Config.n_classes).double()

    def forward(self, input):
        x = self.Wb(input)
        pred = softmax(x)
        return pred

    def fit(self, inputs, labels):
        optimizer = optim.SGD(self.parameters(), lr=self.config.lr)
        self.train()
        losses = []
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            optimizer.zero_grad()

            pred = self(inputs)
            loss = cross_entropy_loss(labels, pred)
            loss.backward()
            optimizer.step()
            average_loss = np.mean(loss.detach().numpy())
            duration = time.time() - start_time
            print 'Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration)
            losses.append(average_loss)
        return losses


def test_softmax_model():
    """Train softmax model for a number of steps."""
    config = Config()

    # Generate random data to train the model on
    np.random.seed(1234)
    inputs = np.random.rand(config.n_samples, config.n_features)
    labels = np.zeros((config.n_samples, config.n_classes), dtype=np.int32)
    labels[:, 0] = 1

    # Build the model and add the variable initializer op
    model = SoftmaxModelPyTorch(config)
    model.apply(init_weights)
    losses = model.fit(inputs, labels)

    # If ops are implemented correctly, the average loss should fall close to zero
    # rapidly.
    assert losses[-1] < .5
    print "Basic (non-exhaustive) classifier tests pass"

if __name__ == "__main__":
    test_softmax_model()
