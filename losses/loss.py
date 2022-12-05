import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .PhysFormerLoss import *


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
        oup = self.mse(y[0], y_hat)
        # fig = plt.figure(1)
        # plt.plot(y[0, :200].cpu().detach().numpy(), '-')
        # plt.plot(y_hat[0, :200].cpu().detach().numpy(), '--')
        # plt.draw()
        # plt.pause(2)
        # plt.close(fig)
        return oup


class MAELoss(nn.Module):
    def __init__(self, subject_number=10):
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


class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self, subject_number):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels, subject):  # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            # if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            # else:
            #    loss += 1 - torch.abs(pearson)

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


# %% Negative Pearson's correlation + Signal-to-Noise-Ratio (NPSNR)

class NPSNR(nn.Module):
    def __init__(self, Lambda, LowF=0.7, upF=3.5, width=0.4):
        super(NPSNR, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Lambda = Lambda
        self.LowF = LowF
        self.upF = upF
        self.width = width
        self.NormaliceK = 1 / 10.9  # Constant to normalize SNR between -1 and 1
        return

    def forward(self, sample: list):
        assert len(sample) >= 3, print('=>[NP_NSNR] ERROR, sample must have 3 values [y_hat, y, time]')
        rppg = sample[0]
        gt = sample[1]
        time = sample[2]
        loss = 0
        for i in range(rppg.shape[0]):
            ##############################
            # PEARSON'S CORRELATION
            ##############################
            sum_x = torch.sum(rppg[i])  # x
            sum_y = torch.sum(gt[i])  # y
            sum_xy = torch.sum(rppg[i] * gt[i])  # xy
            sum_x2 = torch.sum(torch.pow(rppg[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(gt[i], 2))  # y^2
            N = rppg.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
            ##############################
            # SNR
            ##############################
            N = rppg.shape[-1] * 3
            Fs = 1 / time[i].diff().mean()
            freq = torch.arange(0, N, 1, device=self.device) * Fs / N
            fft = torch.abs(torch.fft.fft(rppg[i], dim=-1, n=N)) ** 2
            gt_fft = torch.abs(torch.fft.fft(gt[i], dim=-1, n=N)) ** 2
            fft = fft.masked_fill(torch.logical_or(freq > self.upF, freq < self.LowF).to(self.device), 0)
            gt_fft = gt_fft.masked_fill(torch.logical_or(freq > self.upF, freq < self.LowF).to(self.device), 0)
            PPG_peaksLoc = freq[gt_fft.argmax()]
            mask = torch.zeros(fft.shape[-1], dtype=torch.bool, device=self.device)
            mask = mask.masked_fill(
                torch.logical_and(freq < PPG_peaksLoc + (self.width / 2), PPG_peaksLoc - (self.width / 2) < freq).to(
                    self.device), 1)  # Main signal
            mask = mask.masked_fill(torch.logical_and(freq < PPG_peaksLoc * 2 + (self.width / 2),
                                                      PPG_peaksLoc * 2 - (self.width / 2) < freq).to(self.device),
                                    1)  # Armonic
            power = fft * mask
            noise = fft * mask.logical_not().to(self.device)
            SNR = (10 * torch.log10(power.sum() / noise.sum())) * self.NormaliceK
            ##############################
            # JOIN BOTH LOSS FUNCTION
            ##############################
            loss += 1 - (pearson + (self.Lambda * SNR))

        loss = loss / rppg.shape[0]
        return loss
