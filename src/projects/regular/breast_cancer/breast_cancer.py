import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

sys.path.insert(0, "../../..")
from data.classification.breast_cancer.load import BreastCancerDataset
from src.agents.logistic_regression import LogisticRegAgent


TRAIN_PARAM = {
    "n_epoch": 100
    , "lr": 0.01
    , "eval_metric": "accuracy"
}

DATASET_PARAM = {
    "batch_size": 16
    , "shuffle": True
}


def main(model) -> int:

    data_loader = torch.utils.data.DataLoader(
        BreastCancerDataset("../../../data/classification/breast_cancer/data.csv")
        , **DATASET_PARAM
    )
    
    loss_fun = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=TRAIN_PARAM.get("lr", 0.01))

    print("Epoch\tLoss\t")
    for e in range(TRAIN_PARAM.get("n_epoch", 0)):

        batch_loss = 0
        for i, (batch_x, batch_y) in enumerate(data_loader):

            optimizer.zero_grad()
            pred = model.forward(batch_x)

            loss = loss_fun(pred, batch_y)
            loss.backward()

            batch_loss += loss
            optimizer.step()

        if ((e + 1) % 10 == 0):
            print(f"{e + 1}\t{batch_loss / batch_x.shape[0]}")

    return 0
