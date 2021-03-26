import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    """
    Desc:
        PyTorch implementation of Logistic Regression Model. 
          The predicted values (output of `self.forward(..)`)
          are probabilities.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.linear = nn.Linear(
            self.config.n_factor
            , out_features=1
            , bias=True
        )
        return

    def forward(self, x):
        
        linear = self.linear(x)
        return torch.sigmoid(linear)
