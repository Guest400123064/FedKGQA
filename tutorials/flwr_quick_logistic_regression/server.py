import flwr as fl


CONFIG = {
    "num_rounds": 3
}


if __name__ == "__main__":
    fl.server.start_server(config=CONFIG)
