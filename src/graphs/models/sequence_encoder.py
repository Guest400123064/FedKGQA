import torch
import torch.nn as nn
from easydict import EasyDict


class RNNEncoder(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config

    def forward(self, x):

        pass
