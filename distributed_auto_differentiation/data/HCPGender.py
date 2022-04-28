import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

class HCPGenderDataset(Dataset):
    def __init__(self, label_file='./data/HCP_Gender/labels_HCP_Gender.csv', sessions='./data/HCP_Gender/HCP_AllData_sess1.npz', 
            window_size=10, temporal_size=256, window_stride=10, num_components=100):
        self.sessions = sessions
        self.labels = np.loadtxt(label_file)
        data = np.load(sessions)
        self.temporal_size = temporal_size
        self.window_size = window_size
        self.window_stride = window_stride
        self.num_components = num_components
        samples_per_sub = int(self.temporal_size / self.window_size)
        self.data = np.zeros((data.shape[0], samples_per_sub, data.shape[1], self.window_size))
        for i in range(data.shape[0]):
            for j in range(samples_per_sub):
                 self.data[i, j, :, :] = data[i, :, (j * self.window_stride):(j * self.window_stride) + self.window_size]

    def __getitem__(self, k):
        return torch.from_numpy(self.data[k,...]).float(), torch.Tensor([self.labels[k]]).long().flatten()

    def __len__(self):
        return len(self.labels)

if __name__=="__main__":
    data = HCPGenderDataset()
    print(data[:][0].shape, data[:][1])
    print(len(data))
    