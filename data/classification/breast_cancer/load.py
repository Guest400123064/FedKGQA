import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class BreastCancerDataset(Dataset):

    def __init__(self, path_data="data.csv"):

        # Load raw csv file and recode target (col 1)
        #   M == 1 == Bad
        #   B == 0 == Good
        tmp_df = pd.read_csv(
            path_data
            , header=None
            , index_col=0
        ).replace(
            {1: {'M': 1, 'B': 0}}
        )

        # Split XY & convert to tensors
        self.factor = torch.tensor(
            tmp_df.iloc[:, 1:].values
            , dtype=torch.float32
        )
        self.target = torch.tensor(
            tmp_df.iloc[:, 0].values
            , dtype=torch.long
        )

        # self.source = tmp_df  # DEBUG ONLY
        return

    def __len__(self):

        # Simply # target
        return len(self.target)

    def __getitem__(self, idx):

        # Return the idx'th row (X, y)
        return self.factor[idx], self.target[idx]


if __name__ == "__main__":
    pass
