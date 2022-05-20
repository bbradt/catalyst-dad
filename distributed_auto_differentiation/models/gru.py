import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
GRU_HIDDEN = 64
FC1_OUT = 512
FC2_OUT = 256

class GRU(nn.Module):
    def __init__(
        self, input_shape, n_classes, hidden_dim=GRU_HIDDEN, layer_dim=1, bias=False
    ):
        super(GRU, self).__init__()
        batch, seq_len, n_features = input_shape
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.x2h = nn.Linear(n_features, 3 * hidden_dim, bias=bias)
        self.h2h = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)

        self.flatten = nn.Flatten(1)

        self.fc1 = nn.Linear(seq_len * hidden_dim, FC1_OUT, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(FC1_OUT, FC2_OUT, bias=bias)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(FC2_OUT, n_classes, bias=bias)


    def forward(self, x):
        h0 = Variable(
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        )
        outs = []
        hn = h0[0, :, :]
        for seq in range(x.size(1)):
            xt = x[:, seq, :]
            xt = xt.view(-1, xt.size(1))
            gate_x = self.x2h(xt)
            gate_h = self.h2h(hn)

            gate_x = gate_x.squeeze()
            gate_h = gate_h.squeeze()

            i_r, i_i, i_n = gate_x.chunk(3, 1)
            h_r, h_i, h_n = gate_h.chunk(3, 1)

            resetgate = F.sigmoid(i_r + h_r)
            inputgate = F.sigmoid(i_i + h_i)
            newgate = F.tanh(i_n + (resetgate * h_n))

            hn = newgate + inputgate * (hn - newgate)
            outs.append(hn)

        out = torch.stack(outs, 1)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return torch.softmax(out, 1)