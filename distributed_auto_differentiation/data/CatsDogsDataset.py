"""Module for the CatsDogs dataset - a subset of MNIST"""

from PIL import Image
from torch.utils.data import Dataset

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        """Args:
                file_list: list<str> - list of filenames
                transform: list<Transform> - list of pytorch or catalyst transforms
        """
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        """
        Required overwrite from pytorch Dataset
            returns length of the dataset
        """
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        """Required overwrite from pytorch Dataset
            returns an image and a label given an index
        """
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0

        return img_transformed, label