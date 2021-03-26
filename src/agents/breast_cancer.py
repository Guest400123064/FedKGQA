import os

import torch
import torch.nn as nn

from src.agents.base import BaseAgent
from src.graphs.models.logistic_regression import LogisticRegression
from data.classification.breast_cancer.load import BreastCancerDataLoader


class BreastCancerLRAgent(BaseAgent):

    def __init__(self, config):
        self.config = config

        # Model Init
        self.loader = BreastCancerDataLoader(self.config.data)
        self.model = LogisticRegression(self.config.model)
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters()
            , lr=self.config.train.lr
        )

        # Counter Init
        self.cur_epoch = 0
        self.cur_iter = 0
        return

    def load_checkpoint(self, path_check_pt):

        """
        Desc:
            Check point loader, very useful if training a HUGE model and 
              some exception happened. Not necessarily this implementation.
        """

        pass

    def save_checkpoint(self, path_check_pt="checkpoint.pth.tar", is_best=0):

        """
        Desc:
            Same as stated above.
        """

        pass

    def run(self):

        """
        Desc:
            The entry, like the `main(...)` function. In the main 
              function, we simply call this method with one line of code.
        """

        try:
            self.train()
        except KeyboardInterrupt:
            print("[ INFO. ] :: Keyboard Interruption, session ends")
        return

    def train(self):

        """
        Desc:
            The main worker function that drive the training session.
        """
        for _ in range(self.config.train.max_epoch):
            self.cur_epoch += 1
            self.train_one_epoch()
            self.validate()
        return

    def train_one_epoch(self):

        """
        Desc:
            Helper function of `self.train(...)`. One epoch of training.
        """

        self.model.train()
        self.cur_iter = 0

        for batch_x, batch_y in self.loader.loader_train:
            self.cur_iter += 1

            self.optimizer.zero_grad()
            pred = self.model.forward(batch_x)
            # print(pred)
            loss_val = self.loss_fn(pred, batch_y)
            loss_val.backward()
            self.optimizer.step()

            if self.cur_iter % self.config.train.log_interval == 0:
                print(
                    "[ TRAIN ] :: Epoch: {:}\tBatch: {:}\tBatch Train Loss: {:.6f}".format(
                        self.cur_epoch, self.cur_iter, loss_val.item()
                    )
                )
        return

    def validate(self):

        """
        Desc:
            One cycle of model validation; iterate through all validation 
              samples and calculate (mean) loss
        """

        self.model.eval()
        loss_val, n_batch = 0, 0

        with torch.no_grad():
            for batch_x, batch_y in self.loader.loader_valid:
                n_batch += 1
                pred = self.model.forward(batch_x)
                loss_val += self.loss_fn(pred, batch_y).item()
        print(
            "[ VALID ] :: Epoch: {:}\tBCE Loss: {:.6f}".format(
                self.cur_epoch, loss_val / n_batch
            )
        )
        return loss_val / n_batch

    def finalize(self):

        """
        Desc:
            Finalizes all the operations of the 2 
              Main classes of the process, the operator and the data loader
        """

        pass
