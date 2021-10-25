import torch
import torch.nn as nn
import torch.nn.functional as F

# **Define neural network. I just burrowed from here: https://github.com/pytorch/examples/blob/master/mnist/main.py**
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten(1)
        self.l1 = nn.Linear(784, 2048, bias=True)
        self.mid = nn.Sequential(nn.Linear(2048, 1024, bias=True), nn.BatchNorm1d(1024), nn.ReLU(),
                                 nn.Linear(1024, 512, bias=True), nn.BatchNorm1d(512), nn.ReLU(),
                                 nn.Linear(512, 256, bias=True), nn.BatchNorm1d(256), nn.ReLU()
                                 )
        self.l5 = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.l1(x))
        x = self.mid(x)
        output = F.log_softmax(self.l5(x), dim=1)
        return output
