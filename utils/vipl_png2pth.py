import cv2
import torch
import os
from skimage.util import img_as_float
import scipy.signal
import numpy as np
from ppg_process_common_function import process_pipe, img_process, img_process2
import matplotlib.pyplot as plt


def preprocess_png2pth(path_to_png, path_to_gt_HR, path_to_wave, path_to_time, path_to_save, image_size, length=240):
    # split train and val

    subject = path_to_png.split('/')[-3]
    version = path_to_png.split('/')[-2]
    source = path_to_png.split('/')[-1]
    # if int(version[1:]) in [1, 4]:
    if int(subject[1:]) in range(1, 71):
        save_path = os.path.join(path_to_save, 'train')
    else:
        save_path = os.path.join(path_to_save, 'val')

    # get fps
    with open(path_to_time) as f:
        time_stamp = f.readlines()
        time_stamp = [int(i) for i in time_stamp]
        time_stamp = np.array(time_stamp)
        time_stamp = np.diff(time_stamp)
        fps = 1000./(np.mean(time_stamp))
    f.close()

    # get pngs
    pngs = os.listdir(path_to_png)
    pngs.sort()

    # get GT wave
    with open(path_to_wave) as f:
        gt_wave = f.readlines()
        float_wave = [float(i) for i in gt_wave[1:]]
    f.close()
    # resample
    float_wave_array = np.array(float_wave)
    float_wave_array = scipy.signal.resample_poly(float_wave_array, up=round(fps), down=60)
    float_wave = list(float_wave_array)

    cut_length = int(1.5 * fps)
    # align png and wave
    print(f"pngs length is {len(pngs)}, wave length is {len(float_wave)}, fps is {fps}")
    if len(pngs) > len(float_wave):
        new_float_wave = float_wave.copy()
        a = len(pngs) - len(float_wave)
        for i in range(a):
            index = round(fps) * (i+1)
            if index < len(float_wave):
                value = (float_wave[index-1]+float_wave[index]) / 2
                new_float_wave.insert(index, value)
        float_wave = new_float_wave
    signal_length = min(len(pngs), len(float_wave))
    if signal_length < length - 2 * cut_length:
        return
    pngs = pngs[0:signal_length]
    float_wave = float_wave[0:signal_length]
    print(f"after align, pngs length is {len(pngs)}, wave length is {len(float_wave)}, fps is {fps}")

    # process png and wave
    mt_new, float_wave = process_pipe(float_wave, view=False, output="", name="", fs=fps)

    pngs = pngs[cut_length:-cut_length]

    with open(path_to_gt_HR) as f:
        gt_HR = f.readlines()
        gt_HR = gt_HR[1:]
        gt_HR = [float(i) for i in gt_HR]
        # resample
        float_hr_value = [0]*signal_length

        for hr_index in range(0, len(float_hr_value)):
            if hr_index < len(gt_HR) * round(fps):
                float_hr_value[hr_index] = gt_HR[hr_index//round(fps)]  # set value to every png
            else:
                float_hr_value[hr_index] = float_hr_value[hr_index-1]  # pad
            if float_hr_value[hr_index] > 254:
                print(f"hr value is {float_hr_value[hr_index]}")
                float_hr_value[hr_index] = float_hr_value[hr_index-1]
        float_hr_value = float_hr_value[cut_length:-cut_length]
    f.close()

    # save data
    frame_length = len(pngs)  # subject frame length
    segment_length = length  # time length every input data
    stride = length
    # H = [(输入大小 - 卷积核大小 + 2 * P) / 步长] + 1
    n_segment = (frame_length - segment_length) // stride + 1  # subject segment length

    for i in range(n_segment):
        data = {}
        segment_face = torch.zeros(segment_length, 3, image_size, image_size)
        segment_label = torch.zeros(segment_length, dtype=torch.float32)
        float_label_detrend = np.zeros(segment_length, dtype=float)
        float_hr_value_repeat = np.zeros(segment_length, dtype=float)
        for j in range(i * stride, i * stride + segment_length):
            png_path = os.path.join(path_to_png, pngs[j])
            temp_face = cv2.imread(png_path)
            temp_face = cv2.resize(temp_face, (image_size, image_size))
            temp_face = img_process(temp_face)
            # numpy to tensor
            temp_face = torch.from_numpy(temp_face)
            # (H,W,C) -> (C,H,W)
            temp_face = torch.permute(temp_face, (2, 0, 1))
            segment_face[j - i * stride, :, :, :] = temp_face
            float_label_detrend[j - i * stride] = float_wave[j]
            float_hr_value_repeat[j - i * stride] = float_hr_value[j]
        save_pth_path = save_path + '/' + subject + version + 's' + source[-1] + 'fps' + str(fps) + '_' + str(i) + '.pth'
        segment_face = torch.permute(segment_face, (1, 0, 2, 3))
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
        data['wave'] = (segment_label, int(subject[1:]))
        data['fps'] = fps
        # hr value
        data['value'] = float_hr_value_repeat

        torch.save(data, save_pth_path)


if __name__ == '__main__':
    data_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/VIPL_FACE"
    save_pth_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/VIPL_PTH"
    trackfail_txt = os.path.join(data_dir, 'lighttrack_fail.txt')
    gt_HR_paths = os.path.join(data_dir, 'path_to_gt_HR.txt')
    wave_paths = os.path.join(data_dir, 'path_to_wave.txt')
    time_paths = os.path.join(data_dir, 'path_to_time.txt')

    png_paths = os.path.join(data_dir, 'path_to_png.txt')

    image_size = 128
    frame_length = 160
    with open(gt_HR_paths, 'r') as f_gt:
        gt_HR_list = f_gt.readlines()
    f_gt.close()

    with open(wave_paths, 'r') as f_gt:
        wave_list = f_gt.readlines()
    f_gt.close()

    with open(time_paths, 'r') as f_time:
        time_list = f_time.readlines()
    f_time.close()

    with open(png_paths, 'r') as f_png:
        png_list = f_png.readlines()
    f_png.close()

    with open(trackfail_txt, 'r') as f_trackfail:
        trackfail_list = f_trackfail.readlines()
    f_trackfail.close()

    assert len(gt_HR_list) == len(wave_list)

    print("Start generate pth from pngs ...")
    list_png_gt = zip(png_list, gt_HR_list, wave_list, time_list)
    length = len(gt_HR_list)
    for i, (png_path, gt_HR_path, wave_path, time_path) in enumerate(list_png_gt):
        print(png_path, f"({i + 1}/{length})")
        pngs = os.listdir(png_path.strip())
        if pngs and (gt_HR_path[:-9] + 'video.avi\n') not in trackfail_list and time_path != '\n':  # if pngs is not empyt
            preprocess_png2pth(png_path.strip(), gt_HR_path.strip(), wave_path.strip(), time_path.strip(), save_pth_dir, image_size, frame_length)
        else:
            print(f"Skip {gt_HR_path}")
    print("Generate pth data finsh!")
