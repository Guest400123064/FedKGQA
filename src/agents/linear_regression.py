import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base import BaseAgent


class LinearReg(nn.Module):

    def __init__(self, factor_dim):
        
        super().__init__()
        self.linear = nn.Linear(factor_dim, 1)
        return

    def forward(self, x):
        
        linear = self.linear(x)
        return linear


class LinearRegAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)
