import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from easydict import EasyDict


class WordEmbedding(nn.Module):

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.num_embeddings
            , embedding_dim=config.embedding_dim
        )
        return

    def freeze(self):

        self.embedding.weight.requires_grad = False
        return

    def melt(self):

        self.embedding.weight.requires_grad = True
        return

    def from_pretrained(
        self
        , vocab: Vocab
        , freeze=True
        , padding_idx=None
    ):
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=vocab.vectors
            , freeze=freeze
            , padding_idx=padding_idx
        )
        return

    def forward(self, x):

        return self.embedding(x)
