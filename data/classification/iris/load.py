import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader, dataset, random_split


class IrisDataset(Dataset):

    def __init__(self, path="data.csv"):

        # Load raw csv file and recode target (col 4, start from 0)
        #   Iris-setosa     == 0
        #   Iris-versicolor == 1
        #   Iris-virginica  == 2
        tmp_df = pd.read_csv(
            path, names=[
                "sep_len"
                , "sep_wid"
                , "pet_len"
                , "pet_wid"
                , "class"
            ]
        ).replace(
            {"class": {
                "Iris-setosa": 0
                , "Iris-versicolor": 1
                , "Iris-virginica": 2
            }}
        )

        # Split XY & convert to tensors
        self.factor = torch.tensor(
            tmp_df.loc[:, "sep_len":"pet_wid"].values, dtype=torch.float32
        )
        self.target = torch.tensor(
            tmp_df.loc[:, "class"].values, dtype=torch.long
        )
        self.source = tmp_df  # For Debugging
        return

    def __len__(self):

        # Simply # target
        return len(self.target)

    def __getitem__(self, idx):

        # Return the idx'th row (X, y)
        return self.factor[idx], self.target[idx]


class IrisDataLoader:

    def __init__(self, config):

        # Basic Setup
        self.config = config
        self.dataset = IrisDataset(self.config.path.dataset)

        # Random Partition for Simplicity
        train_len = int(len(self.dataset) * self.config.data.partition[0])
        valid_len = len(self.dataset) - train_len
        train_dataset, valid_dataset = random_split(self.dataset, [train_len, valid_len])

        # Create Loaders
        self.train_loader = DataLoader(
            train_dataset
            , self.config.train.batch_size
            , shuffle=self.config.train.shuffle_train
        )
        self.valid_loader = DataLoader(
            valid_dataset
            , self.config.train.batch_size
            , shuffle=self.config.train.shuffle_valid
        )
        return

    def view_raw(self):

        return self.dataset.source
