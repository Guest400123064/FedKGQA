import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):

    def __init__(self, factor_dim, target_dim):
        super().__init__()
        self.linear = nn.Linear(factor_dim, target_dim)
        return

    def forward(self, x):
        lin = self.linear(x)
        return F.log_softmax(lin, dim=-1)


if __name__ == "__main__":
    pass
