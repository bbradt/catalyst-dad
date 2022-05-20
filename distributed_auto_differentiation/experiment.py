"""
This is the current version of the main experiment runner for distributed AD.

Currently, the only supported experiments involve a FF network with MNIST
and a Visual Transformer with a subset of ImageNet.

Previous experiments from the Arxiv paper can be found in the old version of this
    repository, in which distributed experiments were simulated:
        https://github.com/bbradt/edad
"""

# External Library Imports
import argparse
import os
import json
import numpy as np 
import torch
import torch.nn as nn
from catalyst import dl
from sklearn.model_selection import KFold
import torchvision.transforms as transforms

# Module Imports
from distributed_auto_differentiation.runners import DistributedRunner
from distributed_auto_differentiation.models import get_model
from distributed_auto_differentiation.hooks import ModelHook
from distributed_auto_differentiation.utils import chunks, dprint
from distributed_auto_differentiation.data import get_dataset
from distributed_auto_differentiation.callbacks import BatchTimerCallback
#from distributed_auto_differentiation.criterion.msece import msece

if __name__=="__main__":
    # Argument Parsing
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--name", type=str, default="NONE", help="The name of the experiment. This determines the output directory.")
    argparser.add_argument("--rank", type=int, default=0, help="The rank of this node in the distributed network")
    argparser.add_argument("--num-nodes", type=int, default=1, help="The number of nodes in the distributed network")
    argparser.add_argument("--backend", type=str, default="nccl", help="The pytorch distributed backend to used <nccl/gloo>")
    argparser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:8998", help="The url of the distributed node given as tcp://<URL>:<PORT>")
    argparser.add_argument("--master-port", type=str, default="8998", help="The port to use for distributed training")
    argparser.add_argument("--master-addr", type=str, default="127.0.0.1", help="The URL of the master node")
    argparser.add_argument("--lr", type=float, default=1e-3, help="The learning rate for training")
    argparser.add_argument("--batch-size", type=int, default=16, help="The local batch-size. The effective batch size is the number of sites multiplied by this number.")
    argparser.add_argument("--epochs", type=int, default=10, help="The number of epochs to run")
    argparser.add_argument("--log-dir", type=str, default="logs", help="The catalyst log directory.")
    argparser.add_argument("--distributed-mode", type=str, default="nocomm", help="The type of distributed training to perform: dad, rankdad, or dsgd.")
    argparser.add_argument("--num-folds", type=int, default=5, help="The number of CV folds to support")
    argparser.add_argument("--model", type=str, default="vit", help="The model to use for training - this supports prebuilt models")
    argparser.add_argument("--model-args", type=str, default="[]", help="A list of arguments to send to the model")
    argparser.add_argument("--model-hidden-dims", type=str, default="[128]", help="hidden dims as str")
    argparser.add_argument("--model-kwargs", type=str, default="{}", help="A dictionary string to send kwargs to the model")
    argparser.add_argument("--dataset", type=str, default="catsvsdogs", help="The name of the dataset. Only Mnist and dogsvscats are supported here - for other data see the old repository")
    argparser.add_argument("--criterion", type=str, default="CrossEntropyLoss", help="The pytorch loss function to use <crossentropyloss>")
    argparser.add_argument("--optimizer", type=str, default="Adam", help="the pytorch optimizer to use <adam/sgd>")
    argparser.add_argument("--scheduler", type=str, default="None", help="The learning rate scheduler to use <NOT SUPPORTED>")
    argparser.add_argument("--k", type=int, default=0, help="The fold to use out of the total number of folds must be less than num-folds")
    argparser.add_argument("--N", type=int, default=-1, help="The size of the training data to use. If given as -1, uses the full training set.")
    argparser.add_argument("--pi-numiterations", type=int, default=1, help="The number of power iterations")
    argparser.add_argument("--pi-effective-rank", type=int, default=2, help="The maximum effective rank")
    argparser.add_argument("--pi-use-sigma", type=int, default=1, help="Whether or not to compute sigma")
    argparser.add_argument("--psgd-rank", type=int, default=2, help="Tolerance for power iterations")       
    argparser.add_argument("--pi-tolerance", type=float, default=1e-3, help="Tolerance for power iterations")
    argparser.add_argument('--seed', type=int, default=0)
    abbreviations = {   
        "batch_size": "bs",
        "num_nodes": "nn",
        "backend": "be",
        "epochs": "ep",
        "distributed_mode": "dm",
        "num_folds":"nf",
        "model": "mo",
        "model_hidden_dims": "mh",
        "model_args":"ma",
        "model_kwargs": "mk",
        "dataset": "da",
        "criterion":"cr",
        "scheduler": "sc",
        "k": "k",
        "N": "N",
        "pi_numiterations":"pini",
        "pi_effective_rank":"pief",
        "pi_use_sigma":"pius",
        "pi_tolerance":"pito",
        "psgd_rank":"pwrr",
        "seed":"se"
    }
    ignore_args = ["dist_url", "master_port", "master_addr", "log_dir", "name", "rank", "log_dir"]
    args = argparser.parse_args()

    # Resolve full output directory for site
    mystrs = []
    for k,v in args.__dict__.items():
        if k not in ignore_args:
            if k in abbreviations.keys():
                key = abbreviations[k]
            else:
                key = k
            mystrs.append("%s=%s" % (key ,v))
    fold_name = "-".join(mystrs)
    experiment_dir = os.path.join(args.log_dir, args.name, fold_name, "site_%d" % args.rank)
    os.makedirs(experiment_dir, exist_ok=True)
    dprint("ARGS: ", args.__dict__)
    with open(os.path.join(experiment_dir, "parameters.json"), "w") as file:
        json.dump(args.__dict__, file)
    # initialize distributed process
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    if args.backend == 'nccl':
        os.environ['NCCL_IB_DISABLE'] = '1'

    torch.distributed.init_process_group(backend=args.backend,
                                        init_method=args.dist_url,
                                        world_size=args.num_nodes,
                                        rank=args.rank)

    dprint("the environment variables have been initiaized!")

    # first barrier to coordinate workers and master
    torch.distributed.barrier()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    real_batch = args.batch_size

    # Get dataset, and subset it if desired
    full_data, input_size, num_classes, num_regression = get_dataset(args.dataset)
    if args.N > 0:
        full_data = torch.utils.data.Subset(full_data, range(args.N))
    N = len(full_data)
    nrange = range(N)
    kf = KFold(n_splits=args.num_folds, shuffle=False)
    train_itr = test_itr = None
    for i, (train_itr, test_itr) in enumerate(kf.split(nrange)):
        if i == args.k:
            break
    train_data = torch.utils.data.Subset(full_data, train_itr)


    # Load dataset according to world rank
    model_hidden_dims = json.loads(args.model_hidden_dims)
    model_args = json.loads(args.model_args)
    model_kwargs = json.loads(args.model_kwargs)
    dprint("Model kwargs", model_kwargs)
    model = get_model(args.model, input_size, num_classes + num_regression, *model_args, hidden_dims=model_hidden_dims, **model_kwargs)    
    hook = ModelHook(model, layer_names=["Linear"])

    # Resolution of arguments for loss, optimizer, etc
    if args.criterion.lower() == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion.lower() == "mse":
        criterion = nn.MSELoss()
    elif "msecse" in args.criterion.lower():
        criterion=msece
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)


    # "Split" dataset across sites - this involves sites just grabbing their appropriate subset for now
    n_site = len(train_data)
    nrange_site = range(n_site)
    chunked = list(chunks(nrange_site, args.num_nodes))
    mychunk = chunked[args.rank]
    dprint("Len train full", len(train_data))
    train_data = torch.utils.data.Subset(train_data, mychunk)
    valid_data = torch.utils.data.Subset(full_data, test_itr)
    transform = transforms.Compose(
            [transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data.dataset.transform = transform
    dprint("Len Valid ", len(valid_data))
    dprint("Len Train ", len(train_data))

    # Create dataloaders
    loaders = {
        "train": torch.utils.data.DataLoader(train_data, batch_size=real_batch, shuffle=True, drop_last=True),
        "valid": torch.utils.data.DataLoader(valid_data, batch_size=real_batch, shuffle=True, drop_last=True),
    }

    # Create catalyst runner
    runner = DistributedRunner(**args.__dict__)

    if num_classes > 2:
        callbacks = {
            "accuracy": dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=num_classes),
            "precision-recall": dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=num_classes
            ),
            "auc": dl.AUCCallback(input_key="logits", target_key="targets"),
            "conf": dl.ConfusionMatrixCallback(
                input_key="logits", target_key="targets", num_classes=num_classes
            ),
            "runtime": BatchTimerCallback()
        }
    else:
        callbacks={
            "accuracy": dl.AccuracyCallback(input_key="logits", target_key="targets", num_classes=num_classes),
            "runtime": BatchTimerCallback()
        }

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
        #ddp=True,
        callbacks=callbacks
    )