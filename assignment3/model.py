import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def take_loss(self, labels, preds):
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError

    def train_on_batch(self, optimizer, inputs_batch, labels_batch):
        self.train()
        optimizer.zero_grad()
        train_x = torch.from_numpy(inputs_batch)
        train_y = torch.from_numpy(labels_batch)
        pred = self(train_x)
        loss = self.take_loss(train_y, pred)
        loss.backward()
        optimizer.step()
        loss = np.mean(loss.detach().numpy())
        return loss

    def predict_on_batch(self, inputs_batch):
        self.eval()
        inputs_batch = torch.from_numpy(inputs_batch)
        return np.argmax(self(inputs_batch).detach().numpy(), axis=1)
