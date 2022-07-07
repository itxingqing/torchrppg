import torch
import torch.nn as nn


class Talos(nn.Module):
    def __init__(self):
        super(Talos, self).__init__()
        pass

    def forward(self):
        return None


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y, y_hat):
        oup = self.mse(y, y_hat)
        return oup

