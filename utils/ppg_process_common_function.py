import matplotlib.pyplot as plt
import os

import scipy.signal
from scipy.signal import butter, filtfilt, detrend
from scipy import  signal
import numpy as np
import torch
from scipy.fftpack import fft
import math
from skimage.util import img_as_float
import heartpy

m_avg = lambda t, x, w: (np.asarray([t[i] for i in range(w, len(x) - w)]),
                         np.convolve(x, np.ones((2 * w + 1,)) / (2 * w + 1),
                                     mode='valid'))


def sd(hr):
    return np.std(hr)


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

def xcorr(x):
    length = len(x)
    R = []
    for m in range(0, length):
        sums = 0.0
        for n in range(0, length-m):
            sums = sums + x[n] * x[n+m]
        R.append(sums/length)
    return R


def img_process(img):
    vidLxL = img_as_float(img[:, :, :])  # img_as_float是将图像除以255,变为float型
    vidLxL = vidLxL.astype('float32')
    vidLxL[vidLxL > 1] = 1  # 把数据归一化到1/255～1之间
    vidLxL[vidLxL < (1 / 255)] = 1 / 255  # 把数据归一化到1/255～1之间
    return vidLxL

def img_process2(img):
    img = (img - 127.5) / 128
    return img


def pearson(vector1, vector2):
    n = len(vector1)
    # simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    # sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    # sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    # 分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den

# https://blog.csdn.net/falwat/article/details/127142806
def bpfilter64(s2, fs):

    minfq = 0.8*2/fs
    maxfq = 3*2/fs
    firl_len = round(len(s2)/10)
    bpfilter = scipy.signal.firwin(firl_len, [minfq, maxfq], pass_zero='bandpass')
    s2f = scipy.signal.filtfilt(bpfilter, 1, s2)
    return s2f


def process_pipe(data, view=False, output="", name="", fs=30):
    # moving average
    w_size = int(fs * .5)  # width of moving window
    time = np.linspace(1, len(data), num=len(data))
    mt, ms = m_avg(time, data, w_size)  # computation of moving average

    # remove global modulation
    sign = data[w_size: -w_size] - ms

    # compute signal envelope
    analytical_signal = np.abs(signal.hilbert(sign))

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


def postprocess(ouput, fps, length=240, method='peakdetection'):
    # cal HR use ButterFilt and FouierTransfrom
    # ButterFilt
    if method == 'dft':
        with torch.no_grad():
            ouput_wave = detrend(ouput, type == 'linear')
            [b_pulse, a_pulse] = butter(3, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
            ouput_wave = filtfilt(b_pulse, a_pulse, ouput_wave)
            # FFT
            hr_predict = 0
            time_interp = int(1000/fps)
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
                    time += time_interp

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
    elif method == 'peakdetection':
        wd, m = heartpy.process(ouput, sample_rate=fps, clean_rr=True)
        hr = m['bpm']
    elif method == 'music':
        # https://blog.csdn.net/qq_44628230/article/details/107013924
        R = xcorr(ouput)
        Rx = scipy.linalg.toeplitz(R)
        w, v = np.linalg.eig(Rx)
        v = np.mat(v)
        vh = v.H
        P = []
        N = len(ouput)
        for index in range(0, N+1):
            ew = []
            w = index/N*np.pi
            for k in range(0, N):
                item = complex(np.cos(k*w), np.sin(k*w))
                ew.append(item)
            ew = np.mat(ew)
            ew = ew.T
            eHw = ew.H
            sums = 0
            sums = np.mat(sums)
            for j in range(3, N+1):
                sums = sums + v[:, j-1]*vh[j-1, :]
            fenzi = eHw*sums*ew
            P.append(abs(1/fenzi[0, 0]))
        f = np.linspace(0, N, N+1)/(2*N)
        # plt.title("power spectrum")
        # plt.plot(f, P)
        # plt.xlabel('f')
        # plt.ylabel('P')
        # plt.show()
        bottom = int(0.75*0.5/(fps/2)*(2*N))
        top = int(2.5*0.5/(fps/2)*(2*N))
        f = (np.argmax(P[bottom:top]) + bottom) / (2*N)
        hr = f * fps * 60
    else:
        raise ValueError('method not understood! Needs to be either '
                         '\'peakdetection\', \'dft\' or \'music\' passed: %s' % method)
    return hr


def evaluation(model, path, length=240, visualize=False, method='peakdetection'):
    data = torch.load(path)
    input = data['face']
    input = input[:, 0:length, :, :]
    input = torch.unsqueeze(input, dim=0)
    gt, subject = data['wave']
    gt = gt[0:length]
    fps = data['fps']
    hr_gt = np.mean(data['value'][0:length])
    gt = torch.unsqueeze(gt, dim=0)
    # inference
    input = input.cuda()
    ouput = model(input)
    # cal HR use ButterFilt and FouierTransfrom
    if len(ouput) > 1:
        wave_predict = ouput[0][0, ].cpu().detach().numpy()
    else:
        wave_predict = ouput[0, ].cpu().detach().numpy()
    hr_predict = postprocess(wave_predict, fps=fps, length=length, method=method)
    hr_predict_from_wave_gt = postprocess(gt[0, ].cpu().detach().numpy(), fps=fps, length=length, method=method)
    wave_gt = gt[0, ].cpu().detach().numpy()
    if visualize:
        fig = plt.figure(1)
        plt.plot(wave_predict, '-')
        plt.plot(wave_gt, '--')
        plt.draw()
        plt.pause(2)
        plt.close(fig)

    return hr_predict, hr_gt, wave_predict, wave_gt
