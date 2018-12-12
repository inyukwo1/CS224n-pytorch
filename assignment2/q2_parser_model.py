import cPickle
import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from tqdm import tqdm

from q2_initialization import xavier_weight_init
from utils.parser_utils import minibatches, load_and_preprocess_data


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation. They can then call self.config.<hyperparameter_name> to
    get the hyperparameter settings.
    """
    n_features = 36
    n_classes = 3
    dropout = 0.5  # (p_drop in the handout)
    embed_size = 50
    hidden_size = 200
    batch_size = 1024
    n_epochs = 10
    lr = 0.0005


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.)


class ParserModelPyTorch(nn.Module):
    def __init__(self, config, pretrained_embeddings):
        super(ParserModelPyTorch, self).__init__()

        self.config = config
        self.pretrained_embeddings = nn.Embedding(pretrained_embeddings.shape[0], pretrained_embeddings.shape[1]).float()
        self.pretrained_embeddings.weight = nn.Parameter(torch.from_numpy(pretrained_embeddings).float())
        self.Wb = nn.Linear(Config.embed_size * Config.n_features, Config.hidden_size).float()
        self.Ub = nn.Linear(Config.hidden_size, Config.n_classes).float()

    def forward(self, input):
        x = self.pretrained_embeddings(input).view((-1, Config.n_features * Config.embed_size))
        x = F.relu(self.Wb(x))
        x = F.dropout(x, training=self.training)
        pred = self.Ub(x)
        return pred

    def run_epoch(self, optimizer, parser, train_examples, dev_set):
        losses = []
        for i, (train_x, train_y) in tqdm(enumerate(minibatches(train_examples, self.config.batch_size)), total=1 + len(train_examples) / self.config.batch_size):

            pred = self(train_x)
            loss = torch.sum(-train_y * F.log_softmax(pred, -1), -1).mean()
            loss.backward()
            optimizer.step()
            losses.append(np.mean(loss.detach().numpy()))
        print "loss: {:.3f}".format(np.mean(losses))

        print "Evaluating on dev set",
        dev_UAS, _ = parser.parse(dev_set)
        print "- dev UAS: {:.2f}".format(dev_UAS * 100.0)
        return dev_UAS


    def fit(self, parser, train_examples, dev_set):
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
        print self
        self.train()
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
            dev_UAS = self.run_epoch(optimizer, parser, train_examples, dev_set)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
            print


    def predict_on_batch(self, inputs_batch):
        self.eval()
        inputs_batch = torch.from_numpy(inputs_batch).long()
        return self(inputs_batch).detach().numpy()


def main(debug=True):
    print 80 * "="
    print "INITIALIZING"
    print 80 * "="
    config = Config()
    parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)

    model = ParserModelPyTorch(config, embeddings)
    parser.model = model
    model.apply(init_weights)
    print 80 * "="
    print "Fitting"
    print 80 * "="
    model.fit(parser, train_examples, dev_set)
    print 80 * "="
    print "Evaluating"
    print 80 * "="
    UAS, dependencies = parser.parse(test_set)
    print "- test UAS: {:.2f}".format(UAS * 100.0)
    print "Writing predictions"
    with open('q2_test.predicted.pkl', 'w') as f:
        cPickle.dump(dependencies, f, -1)
    print "Done!"


if __name__ == '__main__':
    main()

