from cgi import test
import sys
import torch
from distributed_auto_differentiation.utils import utils as ut
import os

backend="nccl"
port = "8998"

master = sys.argv[1]
rank = int(sys.argv[2])
world_size = int(sys.argv[3])

os.environ['MASTER_ADDR'] = master
os.environ['MASTER_PORT'] = port
dist_url = "tcp://{master}:{port}".format(master=master, port=port)

torch.distributed.init_process_group(backend=backend,
                                     init_method=dist_url,
                                     world_size=world_size,
                                     rank=rank)
# first barrier to coordinate workers and master
torch.distributed.barrier()
device = "cuda" if torch.cuda.is_available() else "cpu"
if rank > 0:
    test_tensor = (torch.ones(rank+1,rank+1) * rank).to(device)
    print("Here's what my test tensor looks like")
    print(test_tensor)
else:
    test_tensor = None

recv = ut.point_to_master(test_tensor, world_size, device)

print("What did we get...")
for i, received in enumerate(recv):
    print("From rank ", i+1)
    print(received)

if rank == 0:
    test_broad = torch.randn(5,5).to(device)
else:
    test_broad = None
recv = ut.coordinated_broadcast(test_broad, device, 0)
print("Received from broadcast")
print(recv)