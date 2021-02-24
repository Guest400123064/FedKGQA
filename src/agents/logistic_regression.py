import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base import BaseAgent


class LogisticRegression(nn.Module):

    def __init__(self, factor_dim, target_dim):
        
        super().__init__()
        self.linear = nn.Linear(factor_dim, target_dim)
        return

    def forward(self, x):
        
        linear = self.linear(x)
        return F.log_softmax(linear, dim=-1)


class LogisticRegressionAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
