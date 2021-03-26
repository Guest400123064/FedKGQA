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
            print("[ INFO ] :: Keyboard Interruption, session ends")
        return

    def train(self):

        """
        Desc:
            The main worker function that drive the training session.
        """

        raise NotImplementedError

    def train_one_epoch(self):

        """
        Desc:
            Helper function of `self.train(...)`. One epoch of training.
        """

        raise NotImplementedError

    def validate(self):

        """
        Desc:
            One cycle of model validation; iterate through all validation 
              samples and calculate (mean) loss
        """

        pass

    def finalize(self):

        """
        Desc:
            Finalizes all the operations of the 2 
              Main classes of the process, the operator and the data loader
        """

        pass
