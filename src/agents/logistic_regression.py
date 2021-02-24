import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.base import BaseAgent
from src.graphs.models.logistic_regression import LogisticRegres
from src.utils.misc import print_cuda_statistics

class LogisticRegresAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # Model settings
        self.data_loader = None
        self.model = LogisticRegres(self.config)
        self.loss = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters()
            , lr=self.config.lr
        )

        # Initialize Counters
        self.current_episode = 0
        self.current_iteration = 0
        self.episode_durations = []

        # Set CUDA Flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            self.logger.info("Program will run on *****CPU***** ")

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
        pass

    def validate(self):
        pass

    def finalize(self):
        pass
