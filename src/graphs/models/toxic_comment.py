import torch.nn as nn
from easydict import EasyDict

from .sequence_encoder import SequenceEncoder
from .logistic_regression import MultiLogisticRegression


class ToxicComtModel(nn.Module):

    def __init__(self, config: EasyDict):
        super(ToxicComtModel, self).__init__()

        self.seq_encoder = SequenceEncoder(config.sequence_encoder)
        self.output_layer = MultiLogisticRegression(config.output_layer)
        self.config = config
        return

    def forward(self, x):

        """
        Note:
            The input `x` should be sequence(s) of word embeddings
              instead of word indices.
        """

        hid = self.seq_encoder.embed_only(x)
        out = self.output_layer.forward(hid)
        return out
