import os
import sys
import argparse

sys.path.insert(0, "D:\\Users\\lgfz1\\Documents\\fed_kgqa")
from src.utils.config import read_config
from src.agents.iris_regular import IrisLRAgent


def main() -> int:

    _, config = read_config(
        "D:\\Users\\lgfz1\\Documents\\fed_kgqa\\src\\projects\\regular\\iris\\iris.toml"
    )
    print(config)

    agent = IrisLRAgent(config)
    agent.run()

    return 0


if __name__ == "__main__":
    main()

