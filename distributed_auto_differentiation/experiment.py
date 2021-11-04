# Library Imports
import argparse
import json
import os
import time

from catalyst import dl, metrics
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

from distributed_auto_differentiation.data import get_dataset
from distributed_auto_differentiation.hooks import ModelHook
from distributed_auto_differentiation.models import get_model
from distributed_auto_differentiation.utils import chunks, mm_flatten, power_iteration_BC


class CustomRunner(dl.Runner):
    def __init__(self, parsed_args, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.args = parsed_args
        self._logdir = os.path.join(self.args.log_dir, self.args.name)
        self.mode = self.args.distributed_mode.lower()

    def get_engine(self):
        return dl.DistributedDataParallelEngine()

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
            "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
        }

    @property
    def stages(self):
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return self.args.epochs

    def get_loaders(self, stage: str):
        full_data, self._num_classes = get_dataset(self.args.dataset)

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

        if self.engine.is_ddp:
            train_sampler = DistributedSampler(
                train_data,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=True,
            )
            valid_sampler = DistributedSampler(
                valid_data,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=False,
            )
        else:
            train_sampler = valid_sampler = None

        loaders = {
            "train": torch.utils.data.DataLoader(
                train_data,
                batch_size=args.batch_size,
                sampler=train_sampler,
                shuffle=True,
                drop_last=True,
            ),
            "valid": torch.utils.data.DataLoader(
                valid_data,
                batch_size=args.batch_size,
                sampler=valid_sampler,
                shuffle=True,
                drop_last=True,
            ),
        }
        return loaders

    def get_model(self, stage: str):
        model_args = json.loads(self.args.model_args)
        model_kwargs = json.loads(self.args.model_kwargs)
        model = get_model(self.args.model, *model_args, **model_kwargs)
        hook = ModelHook(model, layer_names=["Linear"])
        return model

    def get_criterion(self, stage: str):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, stage: str, model):
        if args.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(params=model.parameters(), lr=self.args.lr)
        elif args.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(params=model.parameters(), lr=self.args.lr)
        return optimizer

    def get_scheduler(self, stage: str, optimizer):
        return None

    def get_callbacks(self, stage: str):
        return {
            "accuracy": dl.AccuracyCallback(
                input_key="logits", target_key="targets", num_classes=self._num_classes
            ),
            "precision-recall": dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=self._num_classes
            ),
            "auc": dl.AUCCallback(input_key="logits", target_key="targets"),
            "conf": dl.ConfusionMatrixCallback(
                input_key="logits", target_key="targets", num_classes=self._num_classes
            ),
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                save_n_best=1,
            ),
            "tqdm": dl.TqdmCallback(),
        }

    def _dad_backward(self):
        hook = self.model.hook
        for key, parameters in zip(hook.keys, hook.parameters):
            input_activations = hook.forward_stats[key]["input"][0]
            deltas = hook.backward_stats[key]["output"][0]
            # print(key, deltas.shape, input_activations.shape, parameters["weight"].grad.shape)
            # print("DELTAS ", key, deltas.shape)
            # print("ACTIVATIONS ", key, input_activations.shape)
            # print("GRAD ", parameters['weight'].grad.data.shape)
            act, delta = mm_flatten(input_activations, deltas)
            act_gathered = [
                torch.zeros_like(act) for _ in range(torch.distributed.get_world_size())
            ]
            delta_gathered = [
                torch.zeros_like(delta) for _ in range(torch.distributed.get_world_size())
            ]

            torch.distributed.all_gather(act_gathered, act)
            torch.distributed.all_gather(delta_gathered, delta)

            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)

            parameters["weight"].grad.data = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            if "bias" in parameters.keys():
                parameters["bias"].grad.data = delta_gathered.sum(0)

    def _rankdad_backward(self):
        hook = self.model.hook
        for key, parameters in zip(hook.keys, hook.parameters):
            input_activations = hook.forward_stats[key]["input"][0]
            deltas = hook.backward_stats[key]["output"][0]
            act, delta = mm_flatten(input_activations, deltas)

            """ Rank reduce with PowerIteration """
            delta_local_reduced, act_local_reduced = power_iteration_BC(
                delta.T,
                act.T,
                # rank=self.reduction_rank,
                # numiterations=self.num_pow_iters,
                device=act.device,
            )
            """Rank-reduction end. """

            """ Pick Max rank of the world and pad to match """
            max_rank = torch.Tensor([delta_local_reduced.shape[1]]).to(self.device)
            torch.distributed.all_reduce(max_rank, torch.distributed.ReduceOp.MAX)

            if max_rank > delta_local_reduced.shape[1]:
                _pad = (0, int(max_rank.item() - delta_local_reduced.shape[1]))
                act_local_reduced = F.pad(act_local_reduced, _pad)
                delta_local_reduced = F.pad(delta_local_reduced, _pad)
            act_gathered = [
                torch.zeros_like(act_local_reduced.T)
                for _ in range(torch.distributed.get_world_size())
            ]
            delta_gathered = [
                torch.zeros_like(delta_local_reduced.T)
                for _ in range(torch.distributed.get_world_size())
            ]

            torch.distributed.all_gather(act_gathered, act_local_reduced.T)
            torch.distributed.all_gather(delta_gathered, delta_local_reduced.T)

            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)

            parameters["weight"].grad.data = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            if "bias" in parameters.keys():
                parameters["bias"].grad.data = delta_gathered.sum(0)

    def _dsgd_backward(self):
        size = torch.distributed.get_world_size()
        for param in self.model.parameters():
            grad_gathered = [torch.zeros_like(param.grad.data) for _ in range(size)]
            torch.distributed.all_gather(grad_gathered, param.grad.data)
            param.grad.data = torch.stack(grad_gathered).sum(0) / float(size)

    def on_loader_start(self, runner):
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss", "cumulative_runtime"]
        }

    def handle_batch(self, batch):
        # print("WHAT IS BATCH", batch)
        input, target = batch
        logits = self.model(input)
        loss = self.criterion(logits, target)
        self.batch_metrics.update({"loss": loss})
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        if self.is_train_loader:
            loss.backward()
            start_time = time.time()
            if self.mode == "dad":
                self._dad_backward()
            elif self.mode == "dsgd":
                self._dsgd_backward()
            elif self.mode == "rankdad":
                self._rankdad_backward()
            runtime = time.time() - start_time
            self.batch_metrics.update({"runtime": runtime})
            self.meters["cumulative_runtime"].update(self.batch_metrics["runtime"], 1)
            self.optimizer.step()
        self.batch = dict(features=input, targets=target, logits=logits)

    def on_loader_end(self, runner):
        for key in ["loss", "cumulative_runtime"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument("--rank", type=int, default=0)
    # argparser.add_argument("--num-nodes", type=int, default=1)
    # argparser.add_argument("--backend", type=str, default="gloo")
    # argparser.add_argument("--dist-url", type=str, default="tcp://127.0.0.1:8998")
    # argparser.add_argument("--master-port", type=str, default="8998")
    # argparser.add_argument("--master-addr", type=str, default="127.0.0.1")
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

    runner = CustomRunner(parsed_args=args)
    runner.run()
