import torch.nn as nn
import torch.nn.functional as F


class RNNLogisticRegres(nn.Module):


    def __init__(self, config):
        
        super().__init__()
        self.config = config
        
