import torch
import torch.nn as nn
from easydict import EasyDict


class RNNEncoder(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config

        self.rnn = nn.RNN(
            input_size=config.input_size    
            , hidden_size=config.hidden_size
            , num_layers=config.num_layers
            , batch_first=True
        )
        return

    def forward(self, x):

        return self.rnn(x)
