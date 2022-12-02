import cv2
import torch
import os
from skimage.util import img_as_float
import numpy as np
from ppg_process_common_function import process_pipe, img_process


def preprocess_png2pth(path_to_png, path_to_gt, path_to_save, image_size):
    # split train and val
    subject = path_to_png.split('/')[-2]
    version = path_to_png.split('/')[-1]
    # if int(version[1:]) in [1, 4]:
    if int(subject[1:]) in [8, 9, 10]:
        save_path = os.path.join(path_to_save, 'val')
    else:
        save_path = os.path.join(path_to_save, 'train')

    # get GT label
    with open(path_to_gt) as f:
        gt = f.readlines()
        gt = gt[1:]
        float_wave = [float(i.split(',')[-1]) for i in gt]
        float_hr_value = [float(i.split(',')[-2]) for i in gt]
        mt_new, float_wave = process_pipe(float_wave, view=False, output="", name="")
        float_hr_value = float_hr_value[45:-45]
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
        save_pth_path = save_path + '/' + subject + version + '_' + str(i) + '.pth'
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
    data_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/P-VHRD-TDM"
    save_pth_dir = "/media/pxierra/e70ff8ce-d5d4-4f52-aa2b-921ff250e5fc/P-VHRD-PTH"
    gt_paths = os.path.join(data_dir, 'path_to_gt.txt')
    png_paths = os.path.join(data_dir, 'path_to_png.txt')
    image_size = 128
    with open(gt_paths, 'r') as f_gt:
        gt_list = f_gt.readlines()
    f_gt.close()

    with open(png_paths, 'r') as f_png:
        png_list = f_png.readlines()
    f_png.close()
    print("Start generate pth from pngs ...")
    list_png_gt = zip(png_list, gt_list)
    length = len(gt_list)
    for i, (png_path, gt_path) in enumerate(list_png_gt):
        print(png_path, f"({i + 1}/{length})")
        preprocess_png2pth(png_path.strip(), gt_path.strip(), save_pth_dir, image_size)
    print("Generate pth data finsh!")
