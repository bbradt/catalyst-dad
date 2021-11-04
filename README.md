# Distributed Auto Differentiation in Catalyst

This code repository is the latest version of the code which is being used to evaluate distributed auto-differentiation for training deep neural networks. 
An [arxiv preprint describing the method can be found here](https://arxiv.org/abs/2102.09631). The code initially used for that paper differs significantly
from the current, evolving codebase, and [can be found here](https://github.com/bbradt/edad). That library only uses Pytorch, and only performs simulated distributed experiments, as a proof of concept that distributed AD can provide equivalent performance. I encourage the committee to focus their attention on this repository, as it is what is being used currently, and what will be used for the future; however, the previous repository is certainly an artifact of the previous research as well.

## Dependencies

First, clone this repository, and CD into the directory:
```
git clone https://github.gatech.edu/bbaker60/dad-catalyst.git;
cd dad-catalyst;
```

This repository has been tested in python version 3.9.7, and using the latest python version is recommended; however, different python 3 versions should work in principle. 

Below are the instructions for creating a new
conda environment for this repository and installing dependencies:

```
conda create --name dad;
conda activate dad;
python -m pip install -r requirements.txt;
```

## Usage

The main file used for running experiments is `distributed_auto_differentiation/experiments.py`. This file also serves as a good example for how the methods of 
distributed auto-differentiation can be applied to different models and data sets. 

Below is the help string which describes all arguments available to this experiment runner:

```
usage: experiment.py [-h] [--rank RANK] [--num-nodes NUM_NODES] [--backend BACKEND] [--dist-url DIST_URL] [--master-port MASTER_PORT]
                     [--master-addr MASTER_ADDR] [--lr LR] [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--name NAME] [--log-dir LOG_DIR]
                     [--distributed-mode DISTRIBUTED_MODE] [--num-folds NUM_FOLDS] [--model MODEL] [--model-args MODEL_ARGS] [--model-kwargs MODEL_KWARGS]
                     [--dataset DATASET] [--criterion CRITERION] [--optimizer OPTIMIZER] [--scheduler SCHEDULER] [--k K] [--N N]

optional arguments:
  -h, --help            show this help message and exit
  --rank RANK           The rank of this node in the distributed network
  --num-nodes NUM_NODES
                        The number of nodes in the distributed network
  --backend BACKEND     The pytorch distributed backend to used <nccl/gloo>
  --dist-url DIST_URL   The url of the distributed node given as tcp://<URL>:<PORT>
  --master-port MASTER_PORT
                        The port to use for distributed training
  --master-addr MASTER_ADDR
                        The URL of the master node
  --lr LR               The learning rate for training
  --batch-size BATCH_SIZE
                        The local batch-size. The effective batch size is the number of sites multiplied by this number.
  --epochs EPOCHS       The number of epochs to run
  --name NAME           The name of the experiment. This determines the output directory.
  --log-dir LOG_DIR     The catalyst log directory.
  --distributed-mode DISTRIBUTED_MODE
                        The type of distributed training to perform: dad, rankdad, or dsgd.
  --num-folds NUM_FOLDS
                        The number of CV folds to support
  --model MODEL         The model to use for training - this supports prebuilt models
  --model-args MODEL_ARGS
                        A list of arguments to send to the model
  --model-kwargs MODEL_KWARGS
                        A dictionary string to send kwargs to the model
  --dataset DATASET     The name of the dataset. Only Mnist and dogsvscats are supported here - for other data see the old repository
  --criterion CRITERION
                        The pytorch loss function to use <crossentropyloss>
  --optimizer OPTIMIZER
                        the pytorch optimizer to use <adam/sgd>
  --scheduler SCHEDULER
                        The learning rate scheduler to use <NOT SUPPORTED>
  --k K                 The fold to use out of the total number of folds must be less than num-folds
  --N N                 The size of the training data to use. If given as -1, uses the full training set.
```

### Two Site MNIST Example

The below example represents how to run the MNIST baseline experiment with **rank-dAD** using two distributed nodes which are connected
over a local network. Suppose we have two machines with IPs 10.0.0.1 and 10.0.0.2 - we will assume that 10.0.0.1 is the
"master" node for our current experiment. 

On node **10.0.0.1**
```
export MASTER_URL=10.0.0.1;
PYTHONPATH=. python distributed_auto_differentiation/experiment.py --name rankdad_mnist --rank 0 --dist-url tcp://${MASTER_URL}:8998 --master-port 8998 --master-addr ${MASTER_URL} --num-nodes 2 --distributed-mode rankdad --model MNISTNet --dataset MNIST
```

On node **10.0.0.2**
```
export MASTER_URL=10.0.0.1;
PYTHONPATH=. python distributed_auto_differentiation/experiment.py --name rankdad_mnist --rank 1 --dist-url tcp://${MASTER_URL}:8998 --master-port 8998 --master-addr ${MASTER_URL} --num-nodes 2 --distributed-mode rankdad --model MNISTNet --dataset MNIST
```

The results will be saved in the directory `logs/rankdad_mnist`, with the subfolder `site_0` corresponding to the master node, and `site_1` corresponding to the 
other node.

The `dad` and `dSGD` methods can be used by changing the flag `--distributed-mode` to `dad` and `dsgd` respectively.

### Two Site Visual Transformer Example

The below example represents how to run the Visual Transformer experiment with **rank-dAD** using two distributed nodes which are connected
over a local network. Suppose we have two machines with IPs 10.0.0.1 and 10.0.0.2 - we will assume that 10.0.0.1 is the
"master" node for our current experiment. 

On node **10.0.0.1**
```
export MASTER_URL=10.0.0.1;
PYTHONPATH=. python distributed_auto_differentiation/experiment.py --name rankdad_vit --rank 0 --dist-url tcp://${MASTER_URL}:8998 --master-port 8998 --master-addr ${MASTER_URL} --num-nodes 2 --distributed-mode rankdad --model vit --dataset dogsvscats
```

On node **10.0.0.2**
```
export MASTER_URL=10.0.0.1;
PYTHONPATH=. python distributed_auto_differentiation/experiment.py --name rankdad_vit --rank 1 --dist-url tcp://${MASTER_URL}:8998 --master-port 8998 --master-addr ${MASTER_URL} --num-nodes 2 --distributed-mode rankdad --model vit --dataset dogsvscats
```

The results will be saved in the directory `logs/rankdad_vit`, with the subfolder `site_0` corresponding to the master node, and `site_1` corresponding to the 
other node.

The `dad` and `dSGD` methods can be used by changing the flag `--distributed-mode` to `dad` and `dsgd` respectively.


## Creating Additional Experiments

Although `experiments.py` only supports two experiments in this repository (again see the previous repository for additional experiments from the Arxiv paper), this 
repository allows for easy additions of other experiments, due in part to the flexibility of Catalyst as a library. 

The below example shows how the MNIST experiment could be created in its own python script. Similar experiments can be performed by substituting in the model, 
datasets, optimizers, and other variables with custom options. 

```
    import torch
    from torch import nn, optim
    from catalyst import dl
    from torchvision import transforms
    from catalyst.contrib.datasets import MNIST
    from distributed_auto_differentiation.runners import DistributedRunner

    model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = MNIST('data', train=True, download=True,
                            transform=transform)
    valid_data = MNIST('data', train=False, download=True,
                            transform=transform)                            
    num_classes = 10

    # Create dataloaders
    loaders = {
        "train": torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True),
        "valid": torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, drop_last=True),
    }

    # Create catalyst runner
    rankdad_runner = DistributedRunner(model,criterion, optimizer, mode="rankdad")

    # run the catalyst experiment
    rankdad_runner.train(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=1,
        logdir="logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,        
        callbacks=[
            dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1, 3, 5)),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
        ],
    )
```

## Repository Design

The `distributed_auto_differentiation` module is organized as follows:

```
    ./__init__.py
    ./experiment.py             # the main runner for new experiments - is in use for measuring runtime, accuracy etc
    ./callbacks                 # contains all custom catalyst callbacks
        __init__.py
        BatchTimerCallback.py   # a custom callback for computing average batch runtime
    ./data                      # contains custom pytorch data set classes
        __init__.py
        CatsDogsDataset.py      # custom class for subset of imagenet
        get_dataset.py          # a wrapper to get any custom datasets, makes experiment.py cleaner
    ./hooks                     # contains all hooks used for interacting with pytorch forward/backward passes
        __init__.py
        ModelHook.py            # a generic, polymorphism-friendly class for grabbing information from forward/backward passes
    ./models                    # contains all custom models used for experiments
        __init__.py
        get_model.py            # a wrapper to grab models, makes experiment.py cleaner
        MNISTNet.py             # a simple, feed-forward network for MNIST
        VisualTransformer.py    # a visual transformer model using Linformer library
    ./runners                   # contains all custom Catalyst runners
        __init__.py
        DistributedRunner.py    # contains the custom runner for distributed AD, rank-dAD and dSGD
                                # all communication methods are wrapped in the same class currently
                                # TODO: perhaps separate classes for each one?        
```

## Description of Computational Artifact

This section is taken verbatim from the report. 

The code used to implement and test dAD and its related methods has gone over several iterations since the initial conception of the method. The primary artifact intended for review by the committee is the newest implementation([https://github.com/bbradt/catalyst-dad](https://github.com/bbradt/catalyst-dad)), which is built with the intention to include the method as a part of the Catalyst accelerated machine learning ecosystem \cite{catalyst}. This newest implementation was created entirely by Bradley Baker, but was reviewed by Aashis Khanal, Sergey Plis, and Sergey Kolesnikov, and the main intention is to use this as a firm foundation for measuring wall-clock runtimes for the different distributed algorithms as well as baselines. This codebase contains the code used to generate figure 10, and contains code to replicate the MNIST experiments, as well as new experiments using a visual trasnformer. This codebase in principal could also be used to replicate figures 2-9; however, the figures from the original Arxiv submission using an older version of the code *([https://github.com/bbradt/edad](https://github.com/bbradt/edad)) have been used in lieu of new figures, as many of those experiments are counted as previous work. 

For distributed learning, the codebase uses the Pytorch wrappers for distributed communication primitives for communication between nodes. As mentioned above, the all-gather and all-reduce, are used. These primatives were required based on the GPU limitations of the pytorch distributed backend. Following the design specifications of [COINSTAC](https://github.com/trendscenter/coinstac), we assume a star network topology, in which a single node is in charge of aggregation and coordination, and other nodes serve as local data-collection sites. 

The code is entirely written in Python, primarily using the [Pytorch](https://pytorch.org/) and [Catalyst](https://github.com/catalyst-team/catalyst) libraries, as well as standard libraries such as Numpy, Scikit-Learn, and Pandas for certain small operations.  The intention with this codebase is to eventually have parts of the code baked into both the Pytorch and Catalyst distributed backends, thus making the methods widely available to the large communities utilizing both ecosystems. In the meantime, the codebase has been created to maximize usability with a wide variety of architectures and experimental setups in both pytorch and catalyst.

The first implementation of distributed AD as a proof of concept can be found here [https://github.com/bbradt/edad](https://github.com/bbradt/edad). This original version implements the idea in a simulated distributed setting, without actual distribution of data between nodes or devices in a network. All subsequent code is based in part off of this original implementation; however, this implementation is no longer maintained as it was intended to show that the methods could provide equiavlent accuracy to the pooled case, rather than testing actual wall-clock runtime. Figures 2-8 were generated using this original codebase. 

A version of the code maintained by Aashis Khanal is included in the COINSTAC toolbox (found here [https://github.com/trendscenter/coinstac-dinunet](https://github.com/trendscenter/coinstac-dinunet)). This implementation  was based off code originally written by both Bradley Baker and Aashis Khanal as well as the writing in the Arxiv preprint.

There are two big limitations to the current implementation. First, all of the code is written in higher-level pytorch to maximize usability. Although many of pytorch's underlying operations for auto-differentiation and distributed learning are implemented in C, we want our method at this stage to be available to developers using pytorch at a high level, and believe that requiring users to fully recompile pytorch with a custom backend solely for our method would significantly limit the scope of usability. Our hope is that we can eventually collaborate with developers on pytorch's backend directly so that they can include the methods in future builds. Although we believe our choice is sensible, it does force our solution to use higher-level workarounds which may affect runtime when compared with the highly-optimized pytorch backend. 

An additional limitation of our current codebase is that we are still working on applying the method to larger-scale experiments. Although we believe our initial smaller-scale demonstration does provide a solid demonstration of the method's benefits, we recognize that the modern distributed deep learning community requires application to very large applications, and comparisons with many baselines in order to make the benefits of the model truly apparent. our first steps toward use with visual transformers on large-scale imaging data does provide a positive step in that direction, but more work is clearly required on that front. 