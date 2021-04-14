import os
import re
import spacy
import torch

import numpy as np
import pandas as pd
from collections import Counter
from easydict import EasyDict

from torchtext.vocab import Vectors, Vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    Dataset
    , DataLoader
    , SubsetRandomSampler
)


class ToxicComtVocab(object):

    def __init__(self, min_freq=5) -> None:
        
        self.is_built = False
        self.min_freq = min_freq
        self.vocab = None
        self.tokenizer = spacy.load("en_core_web_sm")
        return

    def __len__(self):

        ret = -1 if (not self.is_built) else len(self.vocab)
        return ret

    def tokenize(self, text, max_num_char=2000):

        # Chop off long string and tokenize
        text = str(text)[:max_num_char]

        # Preprocess special symbols before tokenization
        text = re.sub(r"[\*\"“”\r\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", text)
        text = re.sub(r"[ ]+", " ", text)
        text = re.sub(r"\!+", "!", text)
        text = re.sub(r"\,+", ",", text)
        text = re.sub(r"\?+", "?", text)

        # Use spaCy tokenizer
        ret = [t.text for t in self.tokenizer(text) if t.text != " "]
        return ret

    def build_vocab(self, list_texts):

        freq = Counter()
        for text in list_texts:
            freq.update(self.tokenize(text))
        
        self.vocab = Vocab(
            counter=freq
            , specials=("<pad>", "<unk>")
            , min_freq=self.min_freq
        )
        self.is_built = True
        return

    def text_to_num(self, text):

        if (not self.is_built):
            raise(Exception("[ ERROR ] :: Vocabulary not built"))

        list_tokens = self.tokenize(text)
        return self.vocab.lookup_indices(list_tokens)

    def load_vectors(self, path_file, path_dir):

        if (not self.is_built):
            raise(Exception("[ ERROR ] :: Vocabulary not built"))

        word_vec = Vectors(path_file, path_dir)
        self.vocab.load_vectors(word_vec)
        return

    def lookup_idx(self, token):

        if (not self.is_built):
            raise(Exception("[ ERROR ] :: Vocabulary not built"))

        return self.vocab.lookup_indices([token])[0]

    def pad_idx(self):

        return self.lookup_idx("<pad>")

    def unk_idx(self):

        return self.lookup_idx("<unk>")


class ToxicComtDataset(Dataset):

    def __init__(self, path):

        # Index:
        #   "id"
        # Predictor/Factor:
        #   "comment_text"
        # Labels/Targets (Binary):
        #   "toxic"
        #   "severe_toxic"
        #   "obscene"
        #   "threat"
        #   "insult"
        #   "identity_hate"
        self.df = pd.read_csv(path, index_col="id")

        # Build vocabulary
        list_comments = self.df.iloc[:, 0].values.tolist()
        self.vocab = ToxicComtVocab()
        self.vocab.build_vocab(list_comments)

        # To Torch tensors
        self.factor = torch.tensor(
            [self.vocab.text_to_num(t) for t in list_comments]
            , dtype=torch.long
        )
        self.target = torch.tensor(
            self.df.iloc[:, 1:].values
            , dtype=torch.float16
        )
        return

    def get_vocab(self):

        return self.vocab

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        return self.factor[idx], self.target[idx]


class ToxicComtCollate(object):

    def __init__(self, idx_pad):
        
        self.idx_pad = idx_pad
        return

    def __call__(self, batch):

        # The following command equates:
        #   factors = [t[0] for t in batch]
        #   targets = [t[1] for t in batch]
        factors, targets = zip(*batch)
        factors = pad_sequence(
            factors
            , batch_first=True
            , padding_value=self.idx_pad
        )
        return factors, targets


class ToxicComtDataLoader(object):

    def __init__(self, config: EasyDict):
        self.config = config

        # Load data
        self.dataset = ToxicComtDataset(
            os.path.join(self.config.path_dir, self.config.path_file)
        )

        # Split train test, possibly dev set
        #   1. Create indices
        #   2. Make samplers
        #   3. Create separate data loaders, feeding both the datset and sampler
        n_sample = len(self.dataset)
        cut_train = int(self.config.pct_train * n_sample)
        idxs_full = np.arange(n_sample)[torch.randperm(n_sample)]  # Shuffle

        self.idxs_train = idxs_full[:cut_train]
        self.idxs_valid = idxs_full[cut_train:]

        splr_train = SubsetRandomSampler(self.idxs_train)
        splr_valid = SubsetRandomSampler(self.idxs_valid)

        # Make loader
        collate_fn = ToxicComtCollate(self.get_vocab().pad_idx())
        self.loader_train = DataLoader(
            self.dataset
            , sampler=splr_train
            , batch_size=self.config.batch_size
            , collate_fn=collate_fn
        )
        self.loader_valid = DataLoader(
            self.dataset
            , sampler=splr_valid
            , batch_size=self.config.batch_size
            , collate_fn=collate_fn
        )
        return

    def get_vocab(self):

        return self.dataset.get_vocab()
