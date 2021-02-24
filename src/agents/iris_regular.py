import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base import BaseAgent
from src.graphs.models.logistic_regression import LogisticRegres
from data.classification.iris.load import IrisDataLoader


class IrisAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # Model settings
        self.data_loader = IrisDataLoader(self.config)
        self.model = LogisticRegres(self.config)
        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr)

        # Initialize Counters
        self.current_episode = 0
        self.current_iteration = 0
        self.episode_durations = []

        # Set CUDA Flag (Skipped for this example for simplicity)
        #   Here we just set device to cpu
        self.device = torch.device("cpu")
        self.logger.info("[ TRAIN ] :: Iris LR model will run on CPU")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, path):
        pass

    def save_checkpoint(self, path, is_best=False):
        pass

    def run(self):
        pass

    def train(self):
        pass

    def train_one_epoch(self):
        
        self.model.train()
        for i, (batch_x, batch_y) in enumerate(self.data_loader.train_loader):

            self.optimizer.zero_grad()
            pred = self.model.forward(batch_x)

            loss = self.loss(pred, batch_y)
            loss.backward()

            self.optimizer.step()
        return

    def validate(self):
        pass

    def finalize(self):
        pass
