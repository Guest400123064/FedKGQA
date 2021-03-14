import os
import re
import spacy
import torch

import numpy as np
import pandas as pd

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    Dataset
    , DataLoader
    , random_split
)


class ToxicComtVocab(object):

    def __init__(self, freq_bar=5):

        self.is_built = False
        self.freq_bar = freq_bar
        self.tokenizer = spacy.load("en_core_web_sm")
        self.itos = {
            0: "<PAD>"    # Padding
            , 1: "<SOS>"  # Start of Sentence
            , 2: "<EOS>"  # End of Sentence
            , 3: "<UNK>"  # Unknown
        }
        self.stoi = {
            "<PAD>": 0
            , "<SOS>": 1
            , "<EOS>": 2
            , "<UNK>": 3
        }
        return

    def __len__(self):
        return len(self.itos)

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

        freq_stat, idx = {}, 4  # start from idx4; idx 3 is <UNK>
        for text in list_texts:
            for token in self.tokenize(text):

                if (token not in freq_stat):
                    freq_stat[token] = 1
                else:
                    freq_stat[token] += 1

                if (freq_stat[token] == self.freq_bar):
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1
        self.is_built = True
        return

    def text_to_num(self, text):

        if (not self.is_built):
            raise(Exception("[ ERROR ] :: Vocabulary not built"))
        try:
            ret = [self.stoi.get(token, self.stoi["<UNK>"]) for token in  self.tokenize(text)]
        except:
            print(f"[ ERROR ] :: Tokenization failed for < {text} >")
        return [self.stoi["<SOS>"]] + ret + [self.stoi["<EOS>"]]
        

class ToxicComtDataset(Dataset):

    def __init__(self, path):

        # Index:
        #   "id"
        # Predictor/Factor:
        #   "comment_text"
        # Labels/Targets:
        #   "toxic"
        #   "severe_toxic"
        #   "obscene"
        #   "threat"
        #   "insult"
        #   "identity_hate"
        self.df = pd.read_csv(path, index_col="id")
        self.factor_raw = self.df["comment_text"]

        # Build a vocabulary
        self.vocab = ToxicComtVocab()
        self.vocab.build_vocab(self.factor_raw)

        # A temporary transformer for numericalize comments
        def text_to_tensor(text):

            ret = self.vocab.text_to_num(text)
            ret = torch.tensor(ret, dtype=torch.long)
            return ret

        # Apply to all comments
        self.factor = self.factor_raw.apply(text_to_tensor)

        # Select just one target for binary classification
        target_col = "toxic"
        self.target = torch.tensor(
            self.df.loc[:, target_col].values, dtype=torch.uint8
        )

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
