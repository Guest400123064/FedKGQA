import torch.nn as nn
import torch.nn.functional as F


class LogisticRegres(nn.Module):

    def __init__(self, config):
        
        super().__init__()
        self.config = config

        self.linear = nn.Linear(
            self.config.factor_dim
            , self.config.target_dim
            , bias=True
        )
        return

    def forward(self, x):
        
        linear = self.linear(x)
        return F.log_softmax(linear, dim=-1)
