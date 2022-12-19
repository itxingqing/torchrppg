import cv2
import torch
import os
from skimage.util import img_as_float
import numpy as np
from ppg_process_common_function import process_pipe, img_process


def preprocess_png2pth(path_to_png, path_to_gt_HR, path_to_wave, path_to_save, image_size):
    # split train and val
    subject = path_to_png.split('/')[-3]
    version = path_to_png.split('/')[-2]
    source = path_to_png.split('/')[-1]
    # if int(version[1:]) in [1, 4]:
    if int(subject[1:]) in range(1, 71):
        save_path = os.path.join(path_to_save, 'train')
    else:
        save_path = os.path.join(path_to_save, 'val')

    if source == 'source1':
        fps = 25
        return
    elif source == 'source4':
        fps = 30
        return
    else:
        fps = 30

    # get pngs
    pngs = os.listdir(path_to_png)
    pngs.sort()

    # get GT label
    with open(path_to_wave) as f:
        gt_wave = f.readlines()
        # resample
        float_wave = [float(i) for i in gt_wave[1::2]]
    f.close()

    cut_length = int(1.5 * fps)
    # align png and wave
    signal_length = min(len(pngs), len(float_wave))
    if signal_length < 240 - 2*cut_length:
        return
    pngs = pngs[0:signal_length]
    float_wave = float_wave[0:signal_length]

    # process png and wave
    mt_new, float_wave = process_pipe(float_wave, view=False, output="", name="", fs=fps)

    pngs = pngs[cut_length:-cut_length]

    with open(path_to_gt_HR) as f:
        gt_HR = f.readlines()
        gt_HR = gt_HR[1:]
        # resample
        float_hr_value = [0]*signal_length

        for hr_index in range(0, len(float_hr_value)):
            if hr_index < len(gt_HR) * fps:
                float_hr_value[hr_index] = gt_HR[hr_index//fps]  # set value to every png
            else:
                float_hr_value[hr_index] = float_hr_value[hr_index-1]  # pad
        float_hr_value = float_hr_value[cut_length:-cut_length]
    f.close()

    # save data
    frame_length = len(pngs)  # subject frame length
    segment_length = 240  # time length every input data
    stride = 240
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
        save_pth_path = save_path + '/' + subject + version + 's' + source[-1] + '_' + str(i) + '.pth'
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
        # hr value
        data['value'] = float_hr_value_repeat

        torch.save(data, save_pth_path)


if __name__ == '__main__':
    data_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/VIPL_FACE"
    save_pth_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/VIPL_PTH"
    trackfail_txt = os.path.join(data_dir, 'lighttrack_fail.txt')
    gt_HR_paths = os.path.join(data_dir, 'path_to_gt_HR.txt')
    wave_paths = os.path.join(data_dir, 'path_to_wave.txt')

    png_paths = os.path.join(data_dir, 'path_to_png.txt')

    image_size = 128
    with open(gt_HR_paths, 'r') as f_gt:
        gt_HR_list = f_gt.readlines()
    f_gt.close()

    with open(wave_paths, 'r') as f_gt:
        wave_list = f_gt.readlines()
    f_gt.close()

    with open(png_paths, 'r') as f_png:
        png_list = f_png.readlines()
    f_png.close()

    with open(trackfail_txt, 'r') as f_trackfail:
        trackfail_list = f_trackfail.readlines()
    f_trackfail.close()

    assert len(gt_HR_list) == len(wave_list)

    print("Start generate pth from pngs ...")
    list_png_gt = zip(png_list, gt_HR_list, wave_list)
    length = len(gt_HR_list)
    for i, (png_path, gt_HR_path, wave_path) in enumerate(list_png_gt):
        print(png_path, f"({i + 1}/{length})")
        pngs = os.listdir(png_path.strip())
        if pngs and (gt_HR_path[:-9] + 'video.avi\n') not in trackfail_list:  # if pngs is not empyt
            preprocess_png2pth(png_path.strip(), gt_HR_path.strip(), wave_path.strip(), save_pth_dir, image_size)
        else:
            print(f"Skip {gt_HR_path}")
    print("Generate pth data finsh!")
