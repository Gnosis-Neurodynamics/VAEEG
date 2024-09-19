import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch


class Dataset_Chunks(Dataset):
    def __init__(self, batch_size=500, pick_start=512, from_DS="train"):
        self.data = np.load("./new_data/%s/whole.npy"%from_DS)
        self.data = self.data[:, pick_start : pick_start + 256]
        samples_num = self.data.shape[0]
        self.chunks_id = np.arange(batch_size, samples_num-batch_size, batch_size)
        self.data = np.split(self.data, self.chunks_id, axis=0)
        print("train data has %d chunks" %len(self.data))
        check_id = [(idx, i) for idx, i in enumerate(self.data) if i.shape[0] == 0]
        assert len(check_id) == 0, "split data error. %s"%check_id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = torch.tensor(self.data[index])
        return x
    