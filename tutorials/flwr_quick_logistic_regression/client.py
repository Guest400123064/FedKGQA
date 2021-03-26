import os, sys
from easydict import EasyDict
from collections import OrderedDict

import numpy as np

import torch
import flwr as fl
from torch.nn.modules import linear

# Agent
sys.path.append("..")
from src.agents.breast_cancer import BreastCancerLRAgent


CONFIG = EasyDict({
    "data": {
        "path_dir": "../data/classification/breast_cancer"
        , "path_file": "data.csv"
        , "pct_train": 0.7
        , "batch_size": 4
    },
    "model": {
        "n_factor": 30
    },
    "train": {
        "max_epoch": 5
        , "lr": 0.01
        , "log_interval": 20
    }
})
bc_agent = BreastCancerLRAgent(CONFIG)


class BreastCancerLRClient(fl.client.NumPyClient):

    def __init__(self) -> None:
        super().__init__()

        self.a = bc_agent
        return

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.a.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.a.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.a.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.a.train_one_epoch()
        return self.get_parameters(), len(self.a.loader.loader_train)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.a.validate()
        return len(self.a.loader.loader_valid), float(loss), float(loss)

x = np.copy(bc_agent.model.linear.weight.detach().numpy())

fl.client.start_numpy_client("127.0.0.1:8080", client=BreastCancerLRClient())

y = bc_agent.model.linear.weight.detach().numpy()

print(np.sum(x - y))
