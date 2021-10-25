from catalyst.contrib.datasets import MNIST
from catalyst.data import ToTensor
from distributed_auto_differentiation.data.CatsDogsDataset import CatsDogsDataset
import os
import glob
from torchvision import transforms

def get_dataset(name, *args, **kwargs):
    if name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNIST('data', train=True, download=True,
                               transform=transform)
    elif name.lower() == "catsvsdogs":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                #transforms.RandomResizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_list = glob.glob(os.path.join("data", "dogsVsCats", "train", '*.jpg'))
        dataset = CatsDogsDataset(train_list, *args, transform=transform, **kwargs)
    return dataset


