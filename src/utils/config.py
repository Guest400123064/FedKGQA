import os
import toml

from easydict import EasyDict


def read_config(path):

    try:
        config_dict = toml.load(path)
        config_easy = EasyDict(config_dict)
        return config_dict, config_easy

    except ValueError:
        print("[ ERROR ] :: Config file parse failed, exit -1")
        exit(-1)

    except FileNotFoundError:
        print("[ ERROR ] :: Config file not found, exit -2")
        exit(-2)


def process_config(path):
    return


def print_config(config):
    return
