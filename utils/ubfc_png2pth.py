import cv2
import torch
import os
from skimage.util import img_as_float
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy import  signal
import matplotlib.pyplot as plt
from ppg_process_common_function import process_pipe


def img_process(img):
    vidLxL = img_as_float(img[:, :, :])  # img_as_float是将图像除以255,变为float型
    vidLxL = vidLxL.astype('float32')
    vidLxL[vidLxL > 1] = 1  # 把数据归一化到1/255～1之间
    vidLxL[vidLxL < (1 / 255)] = 1 / 255  # 把数据归一化到1/255～1之间
    return vidLxL


def preprocess_png2pth(path_to_png, path_to_gt, path_to_save, subject):
    # split train and val
    if int(subject[-2:]) in [1, 4, 5, 8, 9, 10, 11, 12, 13]:
        save_path = os.path.join(path_to_save, 'val')
    else:
        save_path = os.path.join(path_to_save, 'train')

    # get GT label
    with open(path_to_gt) as f:
        gt = f.readlines()
        gtTrace = gt[0].split()
        gtHr = gt[1].split()
        float_wave = [float(i) for i in gtTrace]
        float_hr_value = [float(i) for i in gtHr]
        float_hr_value = float_hr_value[45:-45]
        mt_new, float_wave = process_pipe(float_wave, view=False, output="", name="")
    f.close()

    # save data
    pngs = os.listdir(path_to_png)
    pngs.sort()
    pngs = pngs[45:-45]
    frame_length = len(pngs)  # subject frame length
    segment_length = 240  # time length every input data
    stride = 240
    # H = [(输入大小 - 卷积核大小 + 2 * P) / 步长] + 1
    n_segment = (frame_length - segment_length) // stride + 1  # subject segment length

    for i in range(n_segment):
        data = {}
        segment_face = torch.zeros(segment_length, 3, 128, 128)
        segment_label = torch.zeros(segment_length, dtype=torch.float32)
        float_label_detrend = np.zeros(segment_length, dtype=float)
        float_hr_value_repeat = np.zeros(segment_length, dtype=float)
        for j in range(i * stride, i * stride + segment_length):
            png_path = os.path.join(path_to_png, pngs[j])
            temp_face = cv2.imread(png_path)
            temp_face = cv2.resize(temp_face, (128, 128))
            temp_face = img_process(temp_face)
            # numpy to tensor
            temp_face = torch.from_numpy(temp_face)
            # (H,W,C) -> (C,H,W)
            temp_face = torch.permute(temp_face, (2, 0, 1))
            segment_face[j - i * stride, :, :, :] = temp_face
            float_label_detrend[j - i * stride] = float_wave[j]
            float_hr_value_repeat[j - i * stride] = float_hr_value[j]
        save_pth_path = save_path + '/' + subject + '_' + str(i) + '.pth'
        data['face'] = segment_face
        # normlized wave
        # float_label_detrend = detrend(float_label_detrend, type == 'linear')
        # [b_pulse, a_pulse] = butter(3, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
        # float_label_detrend = filtfilt(b_pulse, a_pulse, float_label_detrend)
        segment_label = torch.from_numpy(float_label_detrend.copy()).float()
        # d_max = segment_label.max()
        # d_min = segment_label.min()
        # segment_label = torch.sub(segment_label, d_min).true_divide(d_max - d_min)
        # segment_label = (segment_label - 0.5).true_divide(0.5)
        data['wave'] = (segment_label, int(subject[-2:]))
        # hr value
        data['value'] = float_hr_value_repeat
        if int(subject[-2:]) not in [11, 18, 24]:
            torch.save(data, save_pth_path)


if __name__ == '__main__':
    dataset_face_dir = "/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/data/pluse/UBFC/TDM_rppg_input/DATASET_2_FACE"
    dataset_gt_dir = "/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/data/pluse/UBFC/DATASET_2"
    save_pth_dir = "/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/data/pluse/UBFC/TDM_rppg_input/DATASET_2_PTH"
    subjects = os.listdir(dataset_face_dir)
    subjects.sort()
    print("Start generate pth from pngs ...")
    length = len(subjects)
    for i, subject in enumerate(subjects):
        print(subject, f"({i+1}/{length})")
        png_dir = os.path.join(dataset_face_dir, subject)
        gt_path = os.path.join(dataset_gt_dir, subject, 'ground_truth.txt')
        preprocess_png2pth(png_dir, gt_path, save_pth_dir, subject)
    print("Generate pth data finsh!")
