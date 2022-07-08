import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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

        # # plt.plot(y[0, 10:200].detach().numpy(), '-')
        # plt.plot(y_hat[0, 10:200].detach().numpy(), '--')
        # plt.show()
        return oup


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, y, y_hat):
        oup = self.mae(y, y_hat)

        # plt.plot(y[0, :].detach().numpy(), '-')
        # plt.plot(y_hat[0, :].detach().numpy(), '--')
        # plt.show()
        return oup

