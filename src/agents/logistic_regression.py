import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base import BaseAgent
from src.graphs.models.logistic_regression import LogisticRegres


class LogisticRegresAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.model = LogisticRegres(self.config)
