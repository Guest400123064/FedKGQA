import torch
import torch.nn as nn
from easydict import EasyDict


class ToxicComtModel(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()

        self.config = config
        self.seq_encoder = CommentEncoder(config.comment_encoder)
        self.out_layer = OutputLayer(config.output_layer)
        return

    def forward(self, x):

        hid = self.seq_encoder(x)
        out = self.out_layer(hid)
        return out


class CommentEncoder(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()

        self.config = config
        self.rnn = nn.RNN(
            config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True
        )
        return

    def forward(self, x):

        _, hid_state = self.rnn(x)
        return hid_state


class OutputLayer(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()

        self.config = config
        self.linear = nn.Linear(
            self.config.n_factor
            , out_features=self.config.n_target
            , bias=True
        )
        return

    def forward(self, x):

        lin = self.linear(x)
        return torch.sigmoid(lin)
