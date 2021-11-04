from catalyst import dl, metrics
import torch
import torch.nn.functional as F
import time

#
from distributed_auto_differentiation.utils import mm_flatten, power_iteration_BC

class DistributedRunner(dl.Runner):
    """This is the distributed Runner for Catalyst, which handles batch activity
            This class also contains the important functions needed for dSGD, dAD, and rank-dAD.
    """
    def __init__(self, model=None, criterion=None, optimizer=None, mode="dsgd", **kwargs):
        """Kwargs:
            model - the pytorch model to run
            criterion - the loss function to use
            optimizer - the pytorch optimizer to use
            mode - the distributed mode to use
            **kwargs - additional kwargs passed to the Runner superclass
        """
        super(DistributedRunner, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.mode = mode.lower()

    def on_loader_start(self, runner):
        """This function initializes measured metrics on dataloader start"""
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss", "cumulative_runtime"]
        }
    def handle_batch(self, batch): 
        """This function handles the batch computation, and is where 
                the appropriate distributed function is selected for sharing statistics. 
                Runtime and other metrics are also evaluated here.
        """
        input, target = batch
        logits = self.model(input)
        loss = self.criterion(logits, target)
        self.batch_metrics.update({"loss": loss})        
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        # Training Only
        if self.is_train_loader:
            loss.backward()
            start_time = time.time()
            # Below is the switch for different distributed AD methods
            if self.mode == "dad":
                self._dad_backward()
            elif self.mode == "dsgd":
                self._dsgd_backward()
            elif self.mode == "rankdad":
                self._rankdad_backward()
            # update batch runtime
            runtime = time.time() - start_time
            self.batch_metrics.update({"runtime": runtime})
            self.meters["cumulative_runtime"].update(self.batch_metrics["runtime"], 1)
            # Step optimizer
            self.optimizer.step()  
        self.batch = dict(features=input, targets=target, logits=logits)

    def _dad_backward(self):
        """This function performs the backwards pass using distributed AD for communication.
            This method is mathematically equivalent to dSGD, and simply sends forward and backward
            stats across the network rather than gradients. 
           In cases where the batch size and number of sites is much smaller than the number of neurons, 
            this avoids sending the larger gradient in favor of these smaller statistics. 
           The full gradient is then computed locally.
        """
        hook = self.model.hook
        for key, parameters in zip(hook.keys, hook.parameters):
            # Grab forward and backward stats from hook
            input_activations = hook.forward_stats[key]['input'][0]
            deltas = hook.backward_stats[key]['output'][0]
            # Flatten matrices prior to computation
            act, delta = mm_flatten(input_activations, deltas)
            act_gathered = [torch.zeros_like(act) for _ in range(torch.distributed.get_world_size())]
            delta_gathered = [torch.zeros_like(delta) for _ in range(torch.distributed.get_world_size())]
            # Communicate Forward and Backward Stats
            torch.distributed.all_gather(act_gathered, act)
            torch.distributed.all_gather(delta_gathered, delta)

            # Concatenate
            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)

            # Product computation of gradient
            parameters['weight'].grad.data = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            if "bias" in parameters.keys():
                parameters["bias"].grad.data = delta_gathered.sum(0)


    def _rankdad_backward(self):
        """This function performs the backwards pass using rank-dAD for communication.
            This method rank-reduces the batch dimension for linear networks (or batch*sequence length
            for recurrent networks). 
           Communication and computation should be efficient when the rank and number of power iterations
            are relatively small.
        """
        hook = self.model.hook
        for key, parameters in zip(hook.keys, hook.parameters):
            input_activations = hook.forward_stats[key]['input'][0]
            deltas = hook.backward_stats[key]['output'][0]
            act, delta = mm_flatten(input_activations, deltas)

            """ Rank reduce with PowerIteration """
            delta_local_reduced, act_local_reduced = power_iteration_BC(
                delta.T,
                act.T,                
                device=act.device
            )
            """Rank-reduction end. """

            """ Pick Max rank of the world and pad to match """
            max_rank = torch.Tensor([delta_local_reduced.shape[1]]).to(self.device)
            torch.distributed.all_reduce(max_rank, torch.distributed.ReduceOp.MAX)

            if max_rank > delta_local_reduced.shape[1]:
                _pad = (0, int(max_rank.item() - delta_local_reduced.shape[1]))
                act_local_reduced = F.pad(act_local_reduced, _pad)
                delta_local_reduced = F.pad(delta_local_reduced, _pad)
            act_gathered = [torch.zeros_like(act_local_reduced.T) for _ in range(torch.distributed.get_world_size())]
            delta_gathered = [torch.zeros_like(delta_local_reduced.T) for _ in range(torch.distributed.get_world_size())]

            torch.distributed.all_gather(act_gathered, act_local_reduced.T)
            torch.distributed.all_gather(delta_gathered, delta_local_reduced.T)

            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)

            parameters['weight'].grad.data = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            if "bias" in parameters.keys():
                parameters["bias"].grad.data = delta_gathered.sum(0)

    def _dsgd_backward(self):
        """This method replicates distributed SGD with all-gather. 
        We use this as a baseline rather than the pytorch native method as a fair 
        comparison, since the native method is implemented in highly optimized C.
        """
        size = torch.distributed.get_world_size()
        for param in self.model.parameters():
            grad_gathered = [torch.zeros_like(param.grad.data) for _ in range(size)]
            torch.distributed.all_gather(grad_gathered, param.grad.data)            
            param.grad.data = torch.stack(grad_gathered).sum(0) / float(size)    

    def on_loader_end(self, runner):
        """This function resolves metrics at the end of runtime"""
        for key in ["loss", "cumulative_runtime"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)