import numpy as np

def chunks(l, n):
    return np.split(np.array(l), n)

def mm_flatten(*tensors):
    if len(tensors[0].shape) > 2:
        dims = list(range(len(tensors[0].shape)))
        return [t.flatten(*dims[:-1]) for t in tensors]
    return tensors
