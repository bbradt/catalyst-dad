from catalyst.contrib.datasets import MNIST
from catalyst.contrib.data import ImageToTensor
from distributed_auto_differentiation.data.CatsDogsDataset import CatsDogsDataset
import os
import glob
from torchvision import transforms


def get_dataset(name, *args, **kwargs):
    """This is a generic getter for datasets, which cleans up the parts of 
            experiments.py which grabs the dataset. This function serves as a
            basic switch function which returns the dataset according to its name.

        Accepted names for this function are:
            "mnist"
            "catsvsdogs"
    """

    if name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNIST('data', train=True, download=True,
                                normalize=(0.1307,0.3081,))
        num_classes = 10
    elif name.lower() == "catsvsdogs" or name.lower() == "dogsvscats":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_list = glob.glob(os.path.join("data", "dogsVsCats", "train", '*.jpg'))
        dataset = CatsDogsDataset(train_list, *args, transform=transform, **kwargs)
        num_classes = 2
    return dataset, num_classes
