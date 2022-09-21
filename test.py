import math
import os

import numpy as np
import torch
from models.model import Model
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, detrend
from scipy.fftpack import fft

if __name__ == '__main__':
    fps = 30
    data_dir = '/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/data/pluse/UBFC/TDM_rppg_input/DATASET_2_PTH/val'
    model_path = '/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/xiongzhuang/1-PycharmProjects/rppg_tdm_talos/saved/models/RPPG_TDM_MSELoss/0919_190645/model_best.pth'
    # load model
    model = Model()
    model = model.to('cuda:0')
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # load data
    data_list = os.listdir(data_dir)
    for data_path in data_list:
        data = torch.load(os.path.join(data_dir, data_path))
        input = data['face']
        input = torch.unsqueeze(input, dim=0)
        gt, subject = data['wave']
        gt = torch.unsqueeze(gt, dim=0)
        # inference
        ouput = model(input)
        fig = plt.figure(1)
        plt.plot(ouput[0, ].cpu().detach().numpy(), '-')
        plt.plot(gt[0, ].cpu().detach().numpy(), '--')
        plt.draw()
        plt.pause(2)
        plt.close(fig)

        # cal HR use ButterFilt and FouierTransfrom
        ## ButterFilt
        ouput_wave = ouput[0, ].cpu().detach().numpy()
        ouput_wave = detrend(ouput_wave, type == 'linear')
        [b_pulse, a_pulse] = butter(3, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
        ouput_wave = filtfilt(b_pulse, a_pulse, ouput_wave)

        gt_wave = gt[0, ].cpu().detach().numpy()
        gt_wave = detrend(gt_wave, type == 'linear')
        [b_pulse, a_pulse] = butter(3, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
        gt_wave = filtfilt(b_pulse, a_pulse, gt_wave)

        fig = plt.figure(2)
        plt.plot(ouput_wave, '-')
        plt.plot(gt_wave, '--')
        plt.draw()
        plt.pause(2)
        plt.close(fig)
        ## FFT
        length = 240
        for index, wave in enumerate([ouput_wave, gt_wave]):
            v_fft = fft(wave)
            v_FA = np.zeros((length, ))
            v_FA[0] = v_fft[0].real * v_fft[0].real
            for i in range(1, int(length/2)):
                v_FA[i] = v_fft[i].real * v_fft[i].real + v_fft[i].imag*v_fft[i].imag
            v_FA[int(length/2)] = v_fft[int(length/2)].real * v_fft[int(length/2)].real

            time = 0.0
            for i in range(0, length-1):
                time += 33

            bottom = (int)(0.7 * time / 1000.0)
            top = (int)(2.5 * time / 1000.0)
            if top > length/2:
                top = length /2
            i_maxpower = 0
            maxpower = 0.0
            for i in range(bottom-2, top-2+1):
                if maxpower < v_FA[i]:
                    maxpower = v_FA[i]
                    i_maxpower = i

            noise_power = 0.0
            signal_power = 0.0
            signal_moment = 0.0
            for i in range(bottom, top+1):
                if (i >= i_maxpower - 2) and (i<=i_maxpower + 2):
                    signal_power += v_FA[i]
                    signal_moment +=i*v_FA[i]
                else:
                    noise_power += v_FA[i]

            snr = 0.0
            if signal_power > 0.01 and noise_power > 0.01:
                snr = 10.0 * math.log10(signal_power/noise_power) + 1
                bias = i_maxpower - (signal_moment / signal_power)
                snr *= (1.0 / 1.0 + bias*bias)

            hr = (signal_moment / signal_power) * 60000.0 / time
            print(f"subject: {subject}")
            print(f"datapath: {data_path}")
            if index == 0:
                print("output", "hr: ", hr, "snr: ", snr)
            else:
                print(" gt", "hr: ", hr, "snr: ", snr, "\n")


