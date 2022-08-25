import cv2
import torch
import os
from skimage.util import img_as_float
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt


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
    f.close()

    # save data
    pngs = os.listdir(path_to_png)
    frame_length = len(pngs)  # subject frame length
    segment_length = 240  # time length every input data
    n_segment = frame_length // segment_length # subject segment length
    pngs.sort()
    for i in range(n_segment):
        data = {}
        segment_face = torch.zeros(segment_length, 3, 36, 36)
        segment_label = torch.zeros(segment_length, dtype=torch.float32)
        float_label_detrend = np.zeros(segment_length, dtype=float)
        for j in range(i*240, i*239+240):
            png_path = os.path.join(path_to_png, pngs[j])
            temp_face = cv2.imread(png_path)
            temp_face = cv2.resize(temp_face, (36, 36))
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
        float_label_detrend = detrend(float_label_detrend, type == 'linear')
        [b_pulse, a_pulse] = butter(3, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
        float_label_detrend = filtfilt(b_pulse, a_pulse, float_label_detrend)
        segment_label = torch.from_numpy(float_label_detrend.copy()).float()
        d_max = segment_label.max()
        d_min = segment_label.min()
        segment_label = torch.sub(segment_label, d_min).true_divide(d_max-d_min)
        segment_label = (segment_label - 0.5).true_divide(0.5)
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
