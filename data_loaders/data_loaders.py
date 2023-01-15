import math
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


class UBFCDataset(Dataset):
    def __init__(self, data_dir, diff_flag=False):
        self.data_dir = data_dir
        self.pth_data = os.listdir(self.data_dir)
        self.diff_flag = diff_flag

    def __len__(self):
        # the number of batches per epoch
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.pth_data[idx])
        data = torch.load(path)
        face_frames = data['face']
        gt_label, subject = data['wave']
        if self.diff_flag:
            last_label = gt_label[-1].unsqueeze(dim=0)
            gt_label = torch.cat([gt_label, last_label], dim=0)
            gt_label = torch.diff(gt_label, dim=0)
            gt_label = gt_label / torch.std(gt_label)
            gt_label[torch.isnan(gt_label)] = 0
        gt_value = data['value']
        if 'fps' not in data.keys():
            fps = 30
        else:
            fps = data['fps']
        gt_value = torch.from_numpy(gt_value.copy()).float()
        return face_frames, gt_label, gt_value, subject, fps


class UBFCDataloader(DataLoader):
    def __init__(self, data_dir, batch_size=32, num_workers=4, shuffle=True, drop_last=True, diff_flag=False):
        self.diff_flag = diff_flag
        self.dataset = UBFCDataset(data_dir, diff_flag=diff_flag)
        super(UBFCDataloader, self).__init__(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)

