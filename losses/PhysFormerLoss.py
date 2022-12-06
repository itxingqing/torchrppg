'''
  modifed based on the HR-CNN
  https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn


# std = 2
def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)


def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    # loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss


def compute_complex_absolute_given_k(output, k, N):
    two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
    hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

    k = k.type(torch.FloatTensor).cuda()
    two_pi_n_over_N = two_pi_n_over_N.cuda()
    hanning = hanning.cuda()

    output = output.view(1, -1) * hanning
    output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
    k = k.view(1, -1, 1)
    two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
    complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                       + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

    return complex_absolute


def function_complex_absolute(output, Fs, bpm_range=None):
    output = output.view(1, -1)

    N = output.size()[1]

    unit_per_hz = Fs / N
    feasible_bpm = bpm_range / 60.0
    k = feasible_bpm / unit_per_hz

    # only calculate feasible PSD range [0.7,4]Hz
    complex_absolute = compute_complex_absolute_given_k(output, k, N)

    return (1.0 / complex_absolute.sum()) * complex_absolute  # Analogous Softmax operator


def cross_entropy_power_spectrum_loss(inputs, target, Fs):
    inputs = inputs.view(1, -1)
    target = target.view(1, -1)
    bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
    # bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

    complex_absolute = function_complex_absolute(inputs, Fs, bpm_range)

    whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
    whole_max_idx = whole_max_idx.type(torch.float)

    # pdb.set_trace()

    # return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
    return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)), torch.abs(
        target[0] - whole_max_idx)


def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
    inputs = inputs.view(1, -1)
    bpm_range = torch.arange(40, 190, dtype=torch.float).cuda()
    # bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
    # bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

    complex_absolute = function_complex_absolute(inputs, Fs, bpm_range)

    whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
    whole_max_idx = whole_max_idx.type(torch.float)

    return whole_max_idx


class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):  # all variable operation
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

            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


def cross_entropy_power_spectrum_DLDL_softmax2(inputs, target, Fs, std):
    target_distribution = [normal_sampling(int(target), i, std) for i in range(140)]
    target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
    target_distribution = torch.Tensor(target_distribution).cuda()

    # pdb.set_trace()

    rank = torch.Tensor([i for i in range(140)]).cuda()

    inputs = inputs.view(1, -1)
    target = target.view(1, -1)

    bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()

    complex_absolute = function_complex_absolute(inputs, Fs, bpm_range)

    fre_distribution = F.softmax(complex_absolute.view(-1))
    loss_distribution_kl = kl_loss(fre_distribution, target_distribution)

    # HR_pre = torch.sum(fre_distribution*rank)

    whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
    whole_max_idx = whole_max_idx.type(torch.float)

    # print("complex_absolute: ", complex_absolute)
    # print("target: ", target)
    # print("complex_absolute: ", complex_absolute.data)
    # print("target: ", target.data)
    a = loss_distribution_kl
    b = F.cross_entropy(complex_absolute, target.view((1)).type(torch.long))
    c = torch.abs(target[0] - whole_max_idx)

    return a, b, c


class PhysFormerLoss(nn.Module):
    def __init__(self, subject_number=10):
        super(PhysFormerLoss, self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, preds, wave, value, subject):
        rPPG, Score1, Score2, Score3 = preds
        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize2

        # Neg_Pearson
        # loss_rPPG = self.criterion_Pearson(rPPG, wave)
        loss_rPPG = 0
        for i in range(rPPG.shape[0]):
            sum_x = torch.sum(rPPG[i])  # x
            sum_y = torch.sum(wave[i])  # y
            sum_xy = torch.sum(rPPG[i] * wave[i])  # xy
            sum_x2 = torch.sum(torch.pow(rPPG[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(wave[i], 2))  # y^2
            N = rPPG.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))

            loss_rPPG += 1 - pearson

        loss_rPPG = loss_rPPG / rPPG.shape[0]

        fre_loss = 0.0
        kl_loss = 0.0
        train_mae = 0.0

        # print("subject：", subject)
        # print("value: ", value)
        value = (value - 40)  # [40, 180]


        for bb in range(rPPG.shape[0]):
            gt_hr_mean = torch.mean(value[bb, :])
            loss_distribution_kl, fre_loss_temp, train_mae_temp = cross_entropy_power_spectrum_DLDL_softmax2(
                rPPG[bb], gt_hr_mean, 30, std=1.0)  # std=1.1
            fre_loss = fre_loss + fre_loss_temp
            kl_loss = kl_loss + loss_distribution_kl
            train_mae = train_mae + train_mae_temp
        fre_loss = fre_loss / rPPG.shape[0]
        kl_loss = kl_loss / rPPG.shape[0]
        train_mae = train_mae / rPPG.shape[0]

        a = 0.1
        b = 1.0
        # print("loss_rPPG：", loss_rPPG.data, 'fre_loss: ', fre_loss.data, 'kl_loss: ', kl_loss.data)

        loss = a*loss_rPPG + b*(fre_loss + kl_loss)
        return loss