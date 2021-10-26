# Library Imports
import argparse
import os
import json
import torch
import torch.nn as nn

from catalyst import dl

#from torchvision import datasets, transforms


from sklearn.model_selection import KFold

# Module Imports
from distributed_auto_differentiation.runners import DistributedRunner
from distributed_auto_differentiation.models import get_model
from distributed_auto_differentiation.hooks import ModelHook
from distributed_auto_differentiation.utils import chunks
from distributed_auto_differentiation.data import get_dataset

# Argument Parsing
argparser = argparse.ArgumentParser()
argparser.add_argument("--rank", type=int, default=0)
argparser.add_argument("--num-nodes", type=int, default=1)
argparser.add_argument("--backend", type=str, default="gloo")
argparser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:8998")
argparser.add_argument("--master-port", type=str, default="8998")
argparser.add_argument("--master-addr", type=str, default="127.0.0.1")
argparser.add_argument("--lr", type=float, default=1e-3)
argparser.add_argument("--batch-size", type=int, default=32)
argparser.add_argument("--epochs", type=int, default=10)
argparser.add_argument("--name", type=str, default="VIT_TEST")
argparser.add_argument("--log-dir", type=str, default="logs")
argparser.add_argument("--distributed-mode", type=str, default="rankdad")
argparser.add_argument("--num-folds", type=int, default=10)
argparser.add_argument("--model", type=str, default="vit")
argparser.add_argument("--model-args", type=str, default="[]")
argparser.add_argument("--model-kwargs", type=str, default="{}")
argparser.add_argument("--dataset", type=str, default="catsvsdogs")
argparser.add_argument("--criterion", type=str, default="CrossEntropyLoss")
argparser.add_argument("--optimizer", type=str, default="Adam")
argparser.add_argument("--scheduler", type=str, default="None")
argparser.add_argument("--k", type=int, default=0)
args = argparser.parse_args()

experiment_dir = os.path.join(args.log_dir, args.name, "site_%d" % args.rank)

os.makedirs(experiment_dir, exist_ok=True)

# initialize distributed process
os.environ['MASTER_ADDR'] = args.master_addr
os.environ['MASTER_PORT'] = args.master_port
torch.distributed.init_process_group(backend=args.backend,
                                     init_method=args.dist_url,
                                     world_size=args.num_nodes,
                                     rank=args.rank)

# first barrier to coordinate workers and master
torch.distributed.barrier()

# Load dataset according to world rank
model_args = json.loads(args.model_args)
model_kwargs = json.loads(args.model_kwargs)
model = get_model(args.model, *model_args, **model_kwargs)
hook = ModelHook(model, layer_names=["Linear"])
if args.criterion.lower() == "crossentropyloss":
    criterion = nn.CrossEntropyLoss()
if args.optimizer.lower() == "adam":
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
elif args.optimizer.lower() == "sgd":
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
scheduler = None
full_data, num_classes = get_dataset(args.dataset)

N = len(full_data)
nrange = range(N)
kf = KFold(n_splits=args.num_folds, shuffle=False)
train_itr = test_itr = None
for i, (train_itr, test_itr) in enumerate(kf.split(nrange)):
    if i == args.k:
        break
train_data = torch.utils.data.Subset(full_data, train_itr)
n_site = len(train_data)
nrange_site = range(n_site)
chunked = list(chunks(nrange_site, args.num_nodes))
mychunk = chunked[args.rank]
train_data = torch.utils.data.Subset(train_data, mychunk)
valid_data = torch.utils.data.Subset(full_data, test_itr)

loaders = {
    "train": torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True),
    "valid": torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, drop_last=True),
}

# Create catalyst runner
runner = DistributedRunner(model,criterion, optimizer, mode=args.distributed_mode)

# run the catalyst experiment
runner.train(
    model=model,
    loaders=loaders,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=args.epochs,
    logdir=experiment_dir,
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
    ddp=True,
    callbacks={
         "accuracy": dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=num_classes),
         "precision-recall": dl.PrecisionRecallF1SupportCallback(
             input_key="logits", target_key="targets", num_classes=num_classes
         ),
         "auc": dl.AUCCallback(input_key="logits", target_key="targets"),
         "conf": dl.ConfusionMatrixCallback(
             input_key="logits", target_key="targets", num_classes=num_classes
         ),
    }
)