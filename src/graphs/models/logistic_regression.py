import torch
import torch.nn as nn
from easydict import EasyDict


class LogisticRegression(nn.Module):

    """
    Desc:
        PyTorch implementation of Logistic Regression Model. 
          The predicted values (output of `self.forward(..)`)
          are probabilities.
    """

    def __init__(self, config: EasyDict):
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


class MultiLogisticRegression(nn.Module):

    """
    Desc:
        PyTorch implementation of Multi Head Logistic Regression Model. 
          The predicted values (output of `self.forward(..)`)
          are probabilities. However, the classes are NOT mutually 
          exclusive, meaning that each channel is a stand-alone 
          LR model. Thus, this is different from the Softmax Regression 
          for multi-class classification.
    """

    def __init__(self, config: EasyDict):
        super().__init__()
        self.config = config

        self.linear = nn.Linear(
            self.config.n_factor
            , out_features=self.config.n_target
            , bias=True
        )
        return

    def forward(self, x):

        linear = self.linear(x)
        return torch.sigmoid(linear)
