from catalyst.contrib.datasets import MNIST
from distributed_auto_differentiation.data import HCPGender
#from catalyst.data import ToTensor
from catalyst.contrib.data import ImageToTensor
from distributed_auto_differentiation.data.CatsDogsDataset import CatsDogsDataset
from distributed_auto_differentiation.data.FSLDataset import FSLDataset
from distributed_auto_differentiation.data.HCPGender import HCPGenderDataset
import os
import glob
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10



def get_dataset(name, *args, **kwargs):
    """This is a generic getter for datasets, which cleans up the parts of 
            experiments.py which grabs the dataset. This function serves as a
            basic switch function which returns the dataset according to its name.

        Accepted names for this function are:
            "mnist"
            "catsvsdogs"
    """
    num_regression = 0
    num_classes = 0
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
    elif name.lower() == "imagenet":
        traindir = '/data/users2/rohib/github/imagenet-data/train'
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
        dataset = ImageFolder(
            traindir,
            transform
        )
        num_classes = 1000
    elif name.lower() == "tiny-imagenet":
        traindir = os.path.join("data", "tiny-imagenet-200", "train")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        )
        dataset = ImageFolder(
            traindir,
            transform
        )
        num_classes = 200
    elif name.lower() == "cifar10":
        transform = transforms.Compose(
            [transforms.Resize((224, 224)),
                transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        num_classes = 10
    elif name.lower() == "fsl_all":
        dataset = FSLDataset("./data/test_fsl/all_data.csv", y_ind=['isControl', 'age'])
        num_classes = 2
        num_regression = 1
    elif name.lower() == "fsl_control":
        dataset = FSLDataset("./data/", y_ind=["isControl"])
        num_classes = 2
    elif name.lower() == "fsl_age":
        dataset = FSLDataset("./data/", y_ind=["age"])
        num_classes = 0
        num_regression = 1
    elif name.lower() == "hcp_gender":
        dataset = HCPGenderDataset()
        num_classes = 2
    return dataset, num_classes, num_regression


