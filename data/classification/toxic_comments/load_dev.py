import os
import re
import spacy
import torch

import pandas as pd
from collections import Counter
from torch._C import PyTorchFileWriter

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


class ToxicComtDataset(Dataset):

    pass


class ToxicComtDataLoader(object):

    def __init__(
        self
        , path_root
        , path_file
        , prob_train=0.7
    ):

        # Get data
        path_full = os.path.join(path_root, path_file)
        self.dataset = ToxicComtDataset(path_full)

        # Random split
        len_train = int(prob_train * len(self.dataset))
        len_valid = len(self.dataset) - len_train
        dataset_train, dataset_valid = random_split(
            self.dataset, [len_train, len_valid]
        )

        # Make loader
        collate_fn = ToxicComtCollate(self.dataset.vocab.stoi.get("<PAD>"))
        self.loader_train = DataLoader(
            dataset_train
            , batch_size=4
            , shuffle=True
            , collate_fn=collate_fn
        )
        self.loader_valid = DataLoader(
            dataset_valid
            , batch_size=4
            , shuffle=True
            , collate_fn=collate_fn
        )
        return

