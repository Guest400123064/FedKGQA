import torch
import torch.nn as nn
from easydict import EasyDict

from .sequence_encoder import RNNEncoder
from .logistic_regression import MultiLogisticRegression


class ToxicComtModel(nn.Module):

    def __init__(self, config: EasyDict) -> None:
        super().__init__()
        self.config = config

        self.config_encoder = None
        self.config_fc_out = None

        self.seq_encoder = None
        self.fc_out = None

        self._init_config()
        self._init_module()
        return

    def _init_config(self):

        # TODO: Implement seq encoder config handle
        self.config_encoder = EasyDict({
            
        })
        self.config_fc_out = EasyDict({
            "n_factor": self.config
            , "n_target": self.config.n_target
        })
        return

    def _init_module(self):

        self.seq_encoder = RNNEncoder(self.config_encoder)
        self.fc_out = MultiLogisticRegression(self.config_fc_out)
        return

    def forward(self, x):

        """
        Note:
            The input `x` should be sequence(s) of word embeddings
              instead of word indices.
        """

        enc = self.seq_encoder(x)
        out = self.fc_out(enc)
        return out
