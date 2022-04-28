import numpy as np
import torch

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

def point_send(source, dst, tensor, device, sub_group):
    #print("Coordinating shape ", source, dst)
    tensor = coordinate_shape(tensor, device, source, sub_group)    
    #print("Sending from, to", source, dst)
    torch.distributed.barrier(sub_group)
    torch.distributed.broadcast(tensor=tensor, src=source, group=sub_group)
    torch.distributed.barrier(sub_group)
    return tensor

def coordinate_shape(tensor, device, src, group, ndim=2):
    #print("Tensor is ", tensor)
    shape = torch.zeros(ndim).to(device)    
    if tensor is not None:
        shape = torch.Tensor(list(tensor.shape)).to(device)
    #print("shape b4", shape)
    torch.distributed.broadcast(tensor=shape, src=src, group=group)    
    torch.distributed.barrier(group)
    #print("shape aft", shape)
    if tensor is None:
        shape = [int(s) for s in shape.tolist()]
        return torch.zeros(*shape).to(device)
    else:
        return tensor

def point_to_master(tensor, world_size, device, master_rank=0):
    rank = torch.distributed.get_rank()
    #print("We are on rank", rank)
    recv = []
    for i in range(world_size):
        if i == master_rank:
            continue        
        sub_group = torch.distributed.new_group([i, master_rank])
        torch.distributed.barrier(sub_group)
        if rank != master_rank:
            recv.append(point_send(i, master_rank, tensor, device, sub_group))                
            #print("Sent Size", recv[-1].size())
        else:
            recv.append(point_send(i, master_rank, tensor, device, sub_group))
            #print("Received Size", recv[-1].size())
        torch.distributed.destroy_process_group(sub_group)
        
    return recv

def coordinated_broadcast(tensor, device, src, group=None):
    tensor = coordinate_shape(tensor, device, src, group)
    torch.distributed.broadcast(tensor=tensor, src=src, group=group)
    torch.distributed.barrier(group)
    return tensor

