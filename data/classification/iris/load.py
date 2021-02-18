import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class IrisDataset(Dataset):

    def __init__(self, path_data="data.csv"):

        # Load raw csv file and recode target (col 4, start from 0)
        #   Iris-setosa     == 0
        #   Iris-versicolor == 1
        #   Iris-virginica  == 2
        tmp_df = pd.read_csv(
            path_data
            , header=None
        ).replace(
            {4: {
                "Iris-setosa": 0
                , "Iris-versicolor": 1
                , "Iris-virginica": 2
            }}
        )

        # Split XY & convert to tensors
        self.factor = torch.tensor(
            tmp_df.iloc[:, 0:3].values
            , dtype=torch.float32
        )
        self.target = torch.tensor(
            tmp_df.iloc[:, 4].values
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
