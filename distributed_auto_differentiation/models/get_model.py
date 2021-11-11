from distributed_auto_differentiation.models.MNISTNet import MNISTNet
from distributed_auto_differentiation.models.VisualTransformer import build_vit


def get_model(name, *args, **kwargs):
    if name.lower() == "mnistnet":
        return MNISTNet(*args, **kwargs)
    elif name.lower() == "vit":
        return build_vit(*args, **kwargs)
