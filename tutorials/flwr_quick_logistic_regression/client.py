# Misc
import os, sys
import numpy as np
from easydict import EasyDict

# Typing
from typing import Dict, List, Tuple

# PyTorch
import torch

# For federated learning using flower
import flwr as fl

# Agent
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../.."
    )
)
from src.agents.breast_cancer import BreastCancerLRAgent


class BreastCancerLRClient(fl.client.NumPyClient):

    def __init__(self, agent) -> None:

        self.agent = agent
        return

    def get_parameters(self) -> List[np.ndarray]:

        params = self.agent.get_parameters()
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:

        self.agent.set_parameters(parameters)
        return

    def fit(
        self
        , parameters: List[np.ndarray]
        , config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int]:

        fit_result = self.agent.fit(parameters, config)
        return fit_result

    def evaluate(
        self
        , parameters: List[np.ndarray]
        , config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, float]]:

        eval_result = self.agent.evaluate(parameters, config)
        return eval_result


if __name__ == "__main__":

    CONFIG = EasyDict({
        "data": {
            "path_dir": "data/classification/breast_cancer"
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
    fl.client.start_numpy_client(
        "127.0.0.1:8080"
        , client=BreastCancerLRClient(bc_agent)
    )
