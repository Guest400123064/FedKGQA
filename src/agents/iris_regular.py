import os
import logging

import torch
import torch.nn as nn

from src.agents.base import BaseAgent
from src.graphs.models.logistic_regression import LogisticRegres
from data.classification.iris.load import IrisDataLoader


class IrisLRAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # Setup logger
        logging.basicConfig(
            filename=os.path.join(
                self.config.path.root_dir
                , self.config.path.train_log
            )
            , filemode="w"
            , level=logging.DEBUG
            , datefmt="%H:%M:%S"
        )
        self.logger = logging.getLogger("IrisLRAgent")

        # Model settings
        self.data_loader = IrisDataLoader(self.config)
        self.model = LogisticRegres(self.config)
        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters()
            , lr=self.config.train.lr
        )

        # Initialize Counters
        self.current_epoch = 0
        self.current_iteration = 0

        # Set CUDA Flag (Skipped for this example for simplicity)
        #   Here we just set device to cpu
        self.device = torch.device("cpu")
        self.logger.info("[ CONFIG ] :: Iris LR model will run on CPU")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.path.checkpoint)
        
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, path):
        pass

    def save_checkpoint(self, path, is_best=False):
        pass

    def run(self):
        
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("[ INFO ] :: Keyboard Interruption, session ends")
        return

    def train(self):
        
        for epoch in range(self.config.train.max_epoch):
            self.train_one_epoch()
            self.validate()
            self.current_epoch += 1
        return

    def train_one_epoch(self):
        
        self.model.train()
        self.current_iteration = 0

        for batch_i, (batch_x, batch_y) in enumerate(self.data_loader.train_loader):
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model.forward(batch_x)

            loss = self.loss(pred, batch_y)
            loss.backward()

            self.optimizer.step()
            self.current_iteration += 1

            batch_i += 1
            if batch_i % self.config.train.log_interval == 0:
                self.logger.info(
                    "[ TRAIN ] :: Epoch {:} [{:}/{:} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                        self.current_epoch
                        , len(batch_x) * batch_i
                        , len(self.data_loader.train_loader.dataset)
                        , 100. * batch_i / len(self.data_loader.train_loader)
                        , loss.item()
                    )
                )
        return

    def validate(self):

        self.model.eval()
        loss, n_correct = 0, 0

        with torch.no_grad():
            for batch_x, batch_y in self.data_loader.valid_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                pred = self.model.forward(batch_x)
                loss += self.loss(pred, batch_y).item()
                
                category = pred.max(1, keepdim=True)[1]
                n_correct += category.eq(batch_y.view_as(category)).sum().item()

        n_sample = len(self.data_loader.valid_loader.dataset)
        loss, accuracy = loss / n_sample, 100. * n_correct / n_sample
        self.logger.info(
            "[ VALID ] :: Epoch {:}\tValid Loss: {:.6f}\t#Correct: [{:}/{:} ({:.0f}%)]".format(
                self.current_epoch
                , loss
                , n_correct
                , n_sample
                , accuracy
            )
        )
        return

    def finalize(self):
        pass
