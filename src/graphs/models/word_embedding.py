import torch
import torch.nn as nn
from easydict import EasyDict


class WordEmbedding(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding()
        return

    def forward(self, x):

        pass
