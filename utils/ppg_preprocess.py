import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, detrend
from scipy import  signal
import numpy as np

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


if __name__ == '__main__':
    data_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/P-VHRD-TDM"
    gt_paths = os.path.join(data_dir, 'path_to_gt.txt')
    with open(gt_paths, 'r') as f_gt:
        gt_list = f_gt.readlines()
    f_gt.close()
    for i, gt_path in enumerate(gt_list):
        gt_path = gt_path.strip()
        with open(gt_path) as f:
            gt = f.readlines()
            gtWave = gt[1:]
            float_label = [float(i.split(',')[-1]) for i in gtWave]
        f.close()
        mt_new, signal_pure = process_pipe(float_label, view=False, output="", name="")
        fig = plt.figure(1)
        plt.plot(signal_pure, '-')
        plt.draw()
        plt.pause(2)
        plt.close(fig)