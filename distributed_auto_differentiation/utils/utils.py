import numpy as np

def chunks(l, n):
    """Split an array l into n chunks, using pigeon-hole principle
        Args: 
            l: list<Object> - a list of objects to split
            n: int - the number of chunks to split
    
    """
    return np.array_split(np.array(l), n)

def mm_flatten(*tensors):
    """A wrapper for flattening tensors in a list so that they may be 
        multiplied using torch.mm
        
        Args:
            *tensors - list of pytorch tensors to be multiplied
    """
    if len(tensors[0].shape) > 2:
        dims = list(range(len(tensors[0].shape)))
        return [t.flatten(*dims[:-1]) for t in tensors]
    return tensors
