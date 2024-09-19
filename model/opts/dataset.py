# -*- coding: utf-8 -*-

import numpy as np
import torch.utils.data
import os


class ClipDataset(torch.utils.data.Dataset):
    """
    Dataset for model training.
    """

    def __init__(self, data_dir, band_name, clip_len=256):
        in_file = os.path.join(data_dir, "%s.npy" % band_name)
        self.data = np.load(in_file)

        self.band = band_name
        self.n_item, self.n_len = self.data.shape
        self.clip_len = clip_len

    def __getitem__(self, index):
        idx = np.random.randint(0, self.n_len - self.clip_len)
        x = self.data[index:index+1, idx: idx + self.clip_len]
        return x

    def __len__(self):
        return self.n_item
