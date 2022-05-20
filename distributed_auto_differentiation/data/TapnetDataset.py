import os
from distributed_auto_differentiation.data.NumpyDataset import NumpyFileDataset
import numpy as np

TRAIN_DATAFILE = "X_train.npy"
TEST_DATAFILE = "X_test.npy"
TRAIN_LABELFILE = "y_train.npy"
TEST_LABELFILE = "y_test.npy"

DATASETS = [
    "ArticularyWordRecognition",
    "AtrialFibrilation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "EigenWorms",
    "Epilepsy",
    "ERing",
    "EthanolConcentration",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PEMS-SF",
    "PenDigits",
    "Phoneme",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
]
DATASET_K = {v.lower(): v for v in DATASETS}


class TapnetDataset(NumpyFileDataset):
    def __init__(self, name, root="./data/tapnet", train=True):
        name = name.replace("tapnet", "")
        name = DATASET_K[name.lower()]
        datapath = os.path.join(root, name)
        if train:
            xf = os.path.join(datapath, TRAIN_DATAFILE)
            yf = os.path.join(datapath, TRAIN_LABELFILE)
        else:
            xf = os.path.join(datapath, TEST_DATAFILE)
            yf = os.path.join(datapath, TEST_LABELFILE)
        super(TapnetDataset, self).__init__(xf, yf, classify=True)
    
    def get_stats(self):
        return (list(self.x[0:1,...].shape), len(np.unique(self.y)))


def get_tapnet_args(name):
    dataset = TapnetDataset(name)
    return (list(dataset.x[0:1,...].shape[1:]), len(np.unique(dataset.y)))
