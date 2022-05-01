from distributed_auto_differentiation.models.MNISTNet import MNISTNet
from distributed_auto_differentiation.models.fsnet import FSNet
from distributed_auto_differentiation.models.VisualTransformer import build_vit
from distributed_auto_differentiation.models.bilstm import ICALstm

def get_model(name, *args, **kwargs):
    if name.lower() == "mnistnet":
        return MNISTNet(*args, **kwargs)
    elif name.lower() == "vit":
        return build_vit(*args, **kwargs)
    elif name.lower() == "vit128":
        return build_vit(*args, dim=128, **kwargs)
    elif name.lower() == "vit256":
        return build_vit(*args, dim=256, **kwargs)
    elif name.lower() == "vit512":
        return build_vit(*args, dim=512, **kwargs)
    elif name.lower() == "vit1024":
        return build_vit(*args, dim=1024, **kwargs)
    elif name.lower() == "vit2048":
        return build_vit(*args, dim=2048, **kwargs)
    elif name.lower() == "vit4096":
        return build_vit(*args, dim=4096, **kwargs)
    elif name.lower() == "fsnet_age":
        return FSNet(66, [256, 128, 64, 32], 1)
    elif name.lower() == "fsnet_all":
        return FSNet(66, [256, 128, 64, 32], 3)
    elif name.lower() == "fsnet_control":
        return FSNet(66, [256, 128, 64, 32], 2)
    elif name.lower() == "icalstm":
        return ICALstm(window_size=10, input_size=256, hidden_size=2048, num_comps=100, num_cls=2, num_layers=1, bidirectional=True)

