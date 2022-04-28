import pandas as pd
from torch.utils.data import Dataset
import os

class FSLDataset(Dataset):
    def __init__(self, csv_file, y_ind=['isControl', 'age']):
        super(FSLDataset, self).__init__()
        self.df = pd.read_csv(csv_file)
        self.y_ind = y_ind

    def __getitem__(self, k):
        row = self.df.iloc[k]
        file = row['freesurferfile']
        df = pd.read_csv(self.path() + os.sep + file, sep='\t', names=['File', file], skiprows=1)
        df = df.set_index(df.columns[0])
        df = df / df.max().astype('float64')
        x = df.T.iloc[0].values
        y = []
        for ynames in self.y_ind:
            y.append(int(row[ynames]))
        if len(y) == 0:
            y = y[0]
        return x, y

    def __len__(self):
        return len(self.df)