import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, detrend
from scipy import  signal
import numpy as np
import torch
from scipy.fftpack import fft
import math

m_avg = lambda t, x, w: (np.asarray([t[i] for i in range(w, len(x) - w)]),
                         np.convolve(x, np.ones((2 * w + 1,)) / (2 * w + 1),
                                     mode='valid'))


def mse(hr, hr_gt):
    hr_zip = zip(hr, hr_gt)
    mse = 0
    for hr, gt in hr_zip:
        mse += pow(hr-gt, 2)
    mse /=len(hr_gt)
    return mse


def rmse(hr, hr_gt):
    return math.sqrt(mse(hr, hr_gt))


def mae(hr, hr_gt):
    hr_zip = zip(hr, hr_gt)
    mae = 0
    for hr, gt in hr_zip:
        mae += abs(hr-gt)
    mae /=len(hr_gt)
    return mae


def process_pipe(data, view=False, output="", name=""):
    fs = 30  # sample rate

    # moving average
    w_size = int(fs * .5)  # width of moving window
    time = np.linspace(1, len(data), num=len(data))
    mt, ms = m_avg(time, data, w_size)  # computation of moving average

    # remove global modulation
    sign = data[w_size: -w_size] - ms

    # compute signal envelope
    analytical_signal = np.abs(signal.hilbert(sign))

    fs = 30
    w_size = int(fs)
    # moving averate of envelope
    mt_new, mov_avg = m_avg(mt, analytical_signal, w_size)

    # remove envelope
    signal_pure = sign[w_size: -w_size] / mov_avg

    if view:
        import matplotlib.pylab as plt

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)
        ax1.plot(time, data, "b-", label="Original")
        ax1.legend(loc='best')
        ax1.set_title("File " + str(name) + " Raw", fontsize=14)  # , fontweight="bold")

        ax2.plot(mt, sign, 'r-', label="Pure signal")
        ax2.plot(mt_new, mov_avg, 'b-', label='Modulation', alpha=.5)
        ax2.legend(loc='best')
        ax2.set_title("Raw -> filtered", fontsize=14)  # , fontweight="bold")

        ax3.plot(mt_new, signal_pure, "g-", label="Demodulated")
        ax3.set_xlim(0, mt[-1])
        ax3.set_title("Raw -> filtered -> demodulated", fontsize=14)  # , fontweight="bold")

        ax3.set_xlabel("Time (sec)", fontsize=14)  # common axis label
        ax3.legend(loc='best')

        fig.tight_layout()
        plt.savefig(output, bbox_inches='tight')

    return mt_new, signal_pure


def evaluation(model, path, fps=30,visualize=False):
    data = torch.load(path)
    input = data['face']
    input = torch.unsqueeze(input, dim=0)
    gt, subject = data['wave']
    hr_gt = np.mean(data['value'])
    gt = torch.unsqueeze(gt, dim=0)
    # inference
    ouput = model(input)
    if visualize:
        fig = plt.figure(1)
        plt.plot(ouput[0, ].cpu().detach().numpy(), '-')
        plt.plot(gt[0, ].cpu().detach().numpy(), '--')
        plt.draw()
        plt.pause(2)
        plt.close(fig)

    # cal HR use ButterFilt and FouierTransfrom
    ## ButterFilt
    ouput_wave = ouput[0,].cpu().detach().numpy()
    ouput_wave = detrend(ouput_wave, type == 'linear')
    [b_pulse, a_pulse] = butter(3, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
    ouput_wave = filtfilt(b_pulse, a_pulse, ouput_wave)

    gt_wave = gt[0,].cpu().detach().numpy()
    gt_wave = detrend(gt_wave, type == 'linear')
    [b_pulse, a_pulse] = butter(3, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
    gt_wave = filtfilt(b_pulse, a_pulse, gt_wave)

    ## FFT
    length = 240
    hr_predict = 0
    # for index, wave in enumerate([ouput_wave, gt_wave]):
    for index, wave in enumerate([ouput_wave]):
        v_fft = fft(wave)
        v_FA = np.zeros((length,))
        v_FA[0] = v_fft[0].real * v_fft[0].real
        for i in range(1, int(length / 2)):
            v_FA[i] = v_fft[i].real * v_fft[i].real + v_fft[i].imag * v_fft[i].imag
        v_FA[int(length / 2)] = v_fft[int(length / 2)].real * v_fft[int(length / 2)].real

        time = 0.0
        for i in range(0, length - 1):
            time += 33

        bottom = (int)(0.7 * time / 1000.0)
        top = (int)(2.5 * time / 1000.0)
        if top > length / 2:
            top = length / 2
        i_maxpower = 0
        maxpower = 0.0
        for i in range(bottom - 2, top - 2 + 1):
            if maxpower < v_FA[i]:
                maxpower = v_FA[i]
                i_maxpower = i

        noise_power = 0.0
        signal_power = 0.0
        signal_moment = 0.0
        for i in range(bottom, top + 1):
            if (i >= i_maxpower - 2) and (i <= i_maxpower + 2):
                signal_power += v_FA[i]
                signal_moment += i * v_FA[i]
            else:
                noise_power += v_FA[i]

        if signal_power > 0.01 and noise_power > 0.01:
            snr = 10.0 * math.log10(signal_power / noise_power) + 1
            bias = i_maxpower - (signal_moment / signal_power)
            snr *= (1.0 / (1.0 + bias * bias))

        hr = (signal_moment / signal_power) * 60000.0 / time
        hr_predict = hr
    return hr_predict, hr_gt