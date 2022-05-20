from distributed_auto_differentiation.models.MNISTNet import MNISTNet
from distributed_auto_differentiation.models.fsnet import FSNet
from distributed_auto_differentiation.models.VisualTransformer import build_vit
from distributed_auto_differentiation.models.bilstm import BiLstm
from distributed_auto_differentiation.models.gru import GRU

def get_model(name, input_size, output_size, *args, hidden_dims=[256,128,64,32], num_layers=1, **kwargs):
    if name.lower() == "fsnet":
        iss = 1
        for ii in input_size:
            iss *= ii
        input_size = iss
        return FSNet(input_size, hidden_dims, output_size)
    elif name.lower() == "gru":
        return GRU(input_size, output_size, hidden_dims[0]) 
    elif name.lower() == "vitclf":
        return build_vit(*args, dim=hidden_dims[0], **kwargs)
    elif name.lower() == "bilstm":
        return BiLstm(window_size=input_size[1], input_size=input_size[-1], hidden_size=hidden_dims[-1], num_comps=1, num_cls=output_size, num_layers=num_layers, bidirectional=True)
    elif name.lower() == "lstm":
        return BiLstm(window_size=input_size[1], input_size=input_size[-1], hidden_size=hidden_dims[-1], num_comps=1, num_cls=output_size, num_layers=num_layers, bidirectional=False)

