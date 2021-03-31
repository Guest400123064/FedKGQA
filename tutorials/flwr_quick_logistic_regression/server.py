from typing import Dict

import flwr as fl
from flwr.server import strategy


def fit_config(rnd: int) -> Dict[str, str]:

    """
    Desc:
        Return a configuration with static batch size and (local) epochs.
    Note:
        The `rnd` (round) parameter is mandatory!
    """

    config = {
        "test_msg": "This is a test message!"
    }
    return config


if __name__ == "__main__":

    # Initialize an aggregation strategy using default FedAvg scheme
    strategy = strategy.FedAvg(
        fraction_fit=0.5,
        min_fit_clients=1,
        min_available_clients=2,
        on_fit_config_fn=fit_config
    )
    fl.server.start_server(config={"num_rounds": 3}, strategy=strategy)
