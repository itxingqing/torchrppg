import math
import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader


class UBFCDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pth_data = os.listdir(self.data_dir)

    def __len__(self):
        # the number of batches per epoch
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        data = torch.load(self.pth_data[idx])
        face_frames = data['face']
        gt_label = data['wave']
        return face_frames, gt_label


class UBFCDataloader(DataLoader):
    def __init__(self, data_dir, batch_size=32, num_workers=4, shuffle=True):
        self.dataset = UBFCDataset(data_dir)
        super(UBFCDataloader, self).__init__(self.dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

