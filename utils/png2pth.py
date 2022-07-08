import cv2
import torch
import os
from skimage.util import img_as_float
from scipy.signal import butter, filtfilt


def img_process(img):
    vidLxL = img_as_float(img[:, :, :])  # img_as_float是将图像除以255,变为float型
    vidLxL = vidLxL.astype('float32')
    vidLxL[vidLxL > 1] = 1  # 把数据归一化到1/255～1之间
    vidLxL[vidLxL < (1 / 255)] = 1 / 255  # 把数据归一化到1/255～1之间
    return vidLxL


def process_face_frame(path_to_png, path_to_gt, path_to_save, subject):
    # split train and val
    if int(subject[-2:]) in [1, 4, 5, 8, 9, 10, 11, 12, 13]:
        save_path = os.path.join(path_to_save, 'train')
    else:
        save_path = os.path.join(path_to_save, 'val')

    # get GT label
    fps = 30
    with open(path_to_gt) as f:
        gt = f.readlines()
        gtTrace = gt[0].split()
        list_ppg = [float(i) for i in gtTrace]
        [b_pulse, a_pulse] = butter(1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
        float_label = filtfilt(b_pulse, a_pulse, list_ppg)
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
        segment_label = torch.zeros(segment_length)
        for j in range(i*240, i*239+240):
            png_path = os.path.join(path_to_png, pngs[j])
            temp_face = cv2.imread(png_path)
            temp_face = img_process(temp_face)
            # numpy to tensor
            temp_face = torch.from_numpy(temp_face)
            # (H,W,C) -> (C,H,W)
            temp_face = torch.permute(temp_face, (2, 0, 1))
            segment_face[j-i*240, :, :, :] = temp_face
            segment_label[j-i*240] = float_label[j]
        save_pth_path = save_path + '/' + subject + '_' + str(i) + '.pth'
        data['face'] = segment_face
        # normlized wave
        data['wave'] = (segment_label - torch.mean(segment_label)) / torch.std(segment_label)
        torch.save(data, save_pth_path)


if __name__ == '__main__':
    dataset_face_dir = "/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/data/pluse/UBFC/TS_CAN_rPPG_input/DATASET_2_FACE"
    dataset_gt_dir = "/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/data/pluse/UBFC/DATASET_2"
    save_pth_dir = "/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/data/pluse/UBFC/TS_CAN_rPPG_input/DATASET_2_PTH"
    subjects = os.listdir(dataset_face_dir)
    subjects.sort()
    for i, subject in enumerate(subjects):
        print(subject)
        png_dir = os.path.join(dataset_face_dir, subject)
        gt_path = os.path.join(dataset_gt_dir, subject, 'ground_truth.txt')
        process_face_frame(png_dir, gt_path, save_pth_dir, subject)
