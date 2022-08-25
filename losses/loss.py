import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Talos(nn.Module):
    def __init__(self, subject_number=10):
        super(Talos, self).__init__()
        self.K = 61
        self.subject_number = subject_number
        self.thea_s = torch.ones((self.subject_number, self.K), requires_grad=True)
        self.thea_s.to('cuda:0')

    def mse(self, yk, y_hat):
        oup = torch.pow(yk-y_hat, 2)
        oup = torch.sum(oup, dim=1) / 240
        return oup

    def forward(self, y_hat, y, subject):
        # y is GT, y_hat is predicted.
        thea_s_exp = torch.exp(self.thea_s)
        sum = torch.unsqueeze(torch.sum(thea_s_exp, dim=1), dim=1)
        sum = sum.repeat((1, self.K))
        p_thea = thea_s_exp / sum

        B, T = y.size()
        k_mse = torch.tensor(0.0)
        for i, k in enumerate(range(-30, 31)):
            # y_pad (B, T) -> (B, T+k)
            if k <= 0:
                y_pad = nn.functional.pad(y, pad=[abs(k), 0], value=0)
                yk = y_pad[:, :T]
            else:
                y_pad = nn.functional.pad(y, pad=[0, k], value=0)
                yk = y_pad[:, -T:]
            # (B, 1) x (1, B)
            mse = self.mse(yk, y_hat).to('cpu')
            select_p_thea = torch.index_select(p_thea[:, i], 0, (subject-1).to('cpu'))
            k_mse = k_mse + torch.sum(mse*select_p_thea) / B
            # k_mse = k_mse + self.mse(yk, y_hat).to('cpu')
        # print("self.thea_s", self.thea_s)
        # print("p_thea", p_thea)
        # fig = plt.figure(1)
        # plt.plot(y_hat[0, :].cpu().detach().numpy(), '-')
        # plt.plot(y[0, :].cpu().detach().numpy(), '--')
        # plt.draw()
        # plt.pause(2)
        # plt.close(fig)
        return k_mse


class MSELoss(nn.Module):
    def __init__(self, subject_number):
        super(MSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y, y_hat, subject):
        oup = self.mse(y, y_hat)
        # fig = plt.figure(1)
        # plt.plot(y[0, :200].cpu().detach().numpy(), '-')
        # plt.plot(y_hat[0, :200].cpu().detach().numpy(), '--')
        # plt.draw()
        # plt.pause(2)
        # plt.close(fig)
        return oup


class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, y, y_hat, subject):
        oup = self.mae(y, y_hat)
        # fig = plt.figure(1)
        # plt.plot(y[0, :].detach().numpy(), '-')
        # plt.plot(y_hat[0, :].detach().numpy(), '--')
        # plt.draw()
        # plt.pause(0.5)
        # plt.close(fig)
        return oup

