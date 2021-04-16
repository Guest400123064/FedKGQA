import torch
import torch.nn as nn
from easydict import EasyDict

from .sequence_encoder import RNNEncoder
from .logistic_regression import MultiLogisticRegression


class ToxicComtModel(nn.Module):

    def __init__(self, config: EasyDict) -> None:
        super().__init__()
        self.config = config

        self.seq_encoder = RNNEncoder(self.config_encoder)
        self.fc_out = MultiLogisticRegression(self.config_fc_out)
        return

    def forward(self, x):

        """
        Note:
            The input `x` should be sequence(s) of word embeddings
              instead of word indices.
        """

        _, hid_out = self.seq_encoder(x)
        out = self.fc_out(hid_out[-1])
        return out
