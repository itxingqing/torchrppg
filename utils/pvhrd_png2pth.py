import cv2
import torch
import os
from skimage.util import img_as_float
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy import  signal
import matplotlib.pyplot as plt


m_avg = lambda t, x, w: (np.asarray([t[i] for i in range(w, len(x) - w)]),
                         np.convolve(x, np.ones((2 * w + 1,)) / (2 * w + 1),
                                     mode='valid'))

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

def img_process(img):
    vidLxL = img_as_float(img[:, :, :])  # img_as_float是将图像除以255,变为float型
    vidLxL = vidLxL.astype('float32')
    vidLxL[vidLxL > 1] = 1  # 把数据归一化到1/255～1之间
    vidLxL[vidLxL < (1 / 255)] = 1 / 255  # 把数据归一化到1/255～1之间
    return vidLxL


def process_face_frame(path_to_png, path_to_gt, path_to_save):
    # split train and val
    subject = path_to_png.split('/')[-2]
    version = path_to_png.split('/')[-1]
    # if int(version[1:]) in [1, 4]:
    if int(subject[1:]) in [8, 9, 10]:
        save_path = os.path.join(path_to_save, 'val')
    else:
        save_path = os.path.join(path_to_save, 'train')

    # get GT label
    fps = 30
    with open(path_to_gt) as f:
        gt = f.readlines()
        gtWave = gt[1:]
        float_label = [float(i.split(',')[-1]) for i in gtWave]
        # float_label = float_label[45:-45]
        mt_new, float_label = process_pipe(float_label, view=False, output="", name="")
    f.close()

    # save data
    pngs = os.listdir(path_to_png)
    pngs.sort()
    pngs = pngs[45:-45]
    frame_length = len(pngs)  # subject frame length
    segment_length = 240  # time length every input data
    n_segment = frame_length // segment_length # subject segment length
    for i in range(n_segment):
        data = {}
        segment_face = torch.zeros(segment_length, 3, 96, 96)
        segment_label = torch.zeros(segment_length, dtype=torch.float32)
        float_label_detrend = np.zeros(segment_length, dtype=float)
        for j in range(i*240, i*239+240):
            png_path = os.path.join(path_to_png, pngs[j])
            temp_face = cv2.imread(png_path)
            temp_face = cv2.resize(temp_face, (96, 96))
            temp_face = img_process(temp_face)
            # numpy to tensor
            temp_face = torch.from_numpy(temp_face)
            # (H,W,C) -> (C,H,W)
            temp_face = torch.permute(temp_face, (2, 0, 1))
            segment_face[j-i*240, :, :, :] = temp_face
            float_label_detrend[j-i*240] = float_label[j]
        save_pth_path = save_path + '/' + subject + version + '_' + str(i) + '.pth'
        data['face'] = segment_face
        # normlized wave
        # float_label_detrend = detrend(float_label_detrend, type == 'linear')
        # [b_pulse, a_pulse] = butter(3, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
        # float_label_detrend = filtfilt(b_pulse, a_pulse, float_label_detrend)
        segment_label = torch.from_numpy(float_label_detrend.copy()).float()
        # d_max = segment_label.max()
        # d_min = segment_label.min()
        # segment_label = torch.sub(segment_label, d_min).true_divide(d_max-d_min)
        # segment_label = (segment_label - 0.5).true_divide(0.5)
        data['wave'] = (segment_label, int(subject[1:]))

        # diff label
        # diff_label = np.diff(segment_label.numpy())
        # diff_label = torch.from_numpy(diff_label)
        # data['face'] = segment_face[1:, :, :, :]
        # data['wave'] = diff_label
        # plt.plot(diff_label[:].detach().numpy(), '--')
        # plt.show()

        torch.save(data, save_pth_path)


if __name__ == '__main__':
    data_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/P-VHRD-TDM"
    save_pth_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/P-VHRD-PTH"
    gt_paths = os.path.join(data_dir, 'path_to_gt.txt')
    png_paths = os.path.join(data_dir, 'path_to_png.txt')
    with open(gt_paths, 'r') as f_gt:
        gt_list = f_gt.readlines()
    f_gt.close()

    with open(png_paths, 'r') as f_png:
        png_list = f_png.readlines()
    f_png.close()
    list_png_gt = zip(png_list, gt_list)
    for i, (png_path, gt_path) in enumerate(list_png_gt):
        process_face_frame(png_path.strip(), gt_path.strip(), save_pth_dir)
