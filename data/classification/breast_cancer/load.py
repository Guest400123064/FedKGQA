import os
import numpy as np
import pandas as pd
from easydict import EasyDict

import torch
from torch.utils.data import (
    Dataset
    , DataLoader
    , SubsetRandomSampler
)



class BreastCancerDataset(Dataset):

    def __init__(self, path):

        # Let's just use pandas to read csv data
        self.df = pd.read_csv(
            path
            , header=None  # This file contains no header
            , index_col=0  # First column is an index column 
        ).replace({

            # Recode the targets such that:
            #   M(alignant) == 1
            #   B(enign) == 0
            1: {'M': 1, 'B': 0}  # 1 denote the second column
        })

        # Split X, Y and convert to tensors
        self.factor = torch.tensor(
            self.df.iloc[:, 1:].values, dtype=torch.float
        ) / 50
        self.target = torch.tensor(
            self.df.iloc[:, :1].values, dtype=torch.float
        )
        return

    def __len__(self):

        # Return # target
        return len(self.target)

    def __getitem__(self, idx):

        # Return a tuple with the first element being the predictors
        return self.factor[idx], self.target[idx]


class BreastCancerDataLoader(object):

    def __init__(self, config):
        self.config = config

        # Load data
        self.dataset = BreastCancerDataset(
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

        self.loader_train = DataLoader(
            self.dataset
            , sampler=splr_train
            , batch_size=self.config.batch_size
        )
        self.loader_valid = DataLoader(
            self.dataset
            , sampler=splr_valid
            , batch_size=self.config.batch_size
        )
        return
