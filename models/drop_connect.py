import torch
from torch import nn

class DropConnect(nn.Module):
    def __init__(self, p):
        super(DropConnect, self).__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.empty(x.size()).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)
