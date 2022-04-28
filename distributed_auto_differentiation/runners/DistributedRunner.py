from catalyst import dl, metrics
import torch
import torch.nn.functional as F
import time
import inspect

#
from distributed_auto_differentiation.utils import mm_flatten, power_iteration_BC, point_to_master, coordinated_broadcast

def invalidArgs(func, argdict):
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if varkw: return set()  # All accepted
    return set(argdict) - set(varkw)

class DistributedRunner(dl.Runner):
    """This is the distributed Runner for Catalyst, which handles batch activity
            This class also contains the important functions needed for dSGD, dAD, and rank-dAD.
    """
    def __init__(self, model=None, criterion=None, optimizer=None, mode="dsgd", rank=None, **kwargs):
        """Kwargs:
            model - the pytorch model to run
            criterion - the loss function to use
            optimizer - the pytorch optimizer to use
            mode - the distributed mode to use
            **kwargs - additional kwargs passed to the Runner superclass
        """
        #invalids = invalidArgs(super(DistributedRunner, self).__init__, kwargs)
        #valid_kwargs = {k:v for k, v in kwargs.items() if k not in invalids}
        super(DistributedRunner, self).__init__()
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.kwargs = kwargs
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
        try:
            logits, _ = self.model(input)
        except Exception as e:
            logits = self.model(input)
        #print(logits.shape, target.shape)
        #print(logits, target.flatten())
        loss = self.criterion(logits, target.flatten())
        self.batch_metrics.update({"loss": loss})        
        for key in ["loss"]:
            self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        # Training Only
        if self.is_train_loader:
            loss.backward()
            start_time = time.time()
            # Below is the switch for different distributed AD methods
            if self.mode == "dad":
                comm = self._dad_backward(**self.kwargs)
            elif self.mode == "dsgd":
                comm = self._dsgd_backward(**self.kwargs)
            elif self.mode == "rankdad_ar":
                comm = self._rankdad_allgather_backward(**self.kwargs)
            elif self.mode == "rankdad":
                comm = self._rankdad_p2p_backward(**self.kwargs)
            elif self.mode == "topk":
                comm = self._topk_backward(**self.kwargs)
            # update batch runtime
            runtime = time.time() - start_time
            self.batch_metrics.update({"runtime": runtime})
            self.meters["cumulative_runtime"].update(self.batch_metrics["runtime"], 1)
            self.batch_metrics.update({"bits": runtime})
            # Step optimizer
            self.optimizer.step()  
        self.batch = dict(features=input, targets=target, logits=logits)

    def _dad_backward(self,**kwargs):
        """This function performs the backwards pass using distributed AD for communication.
            This method is mathematically equivalent to dSGD, and simply sends forward and backward
            stats across the network rather than gradients. 
           In cases where the batch size and number of sites is much smaller than the number of neurons, 
            this avoids sending the larger gradient in favor of these smaller statistics. 
           The full gradient is then computed locally.
        """
        hook = self.model.hook
        comm = 0
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
            for x in act_gathered:
                comm += x.element_size() * x.nelement()
            for x in delta_gathered:
                comm += x.element_size() * x.nelement()
            # Concatenate
            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)

            # Product computation of gradient
            parameters['weight'].grad.data = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            if "bias" in parameters.keys():
                parameters["bias"].grad.data = delta_gathered.sum(0)
        return comm


    def _rankdad_allgather_backward(self, pi_effective_rank=3, pi_numiterations=1, pi_use_sigma=True, **kwargs):
        """This function performs the backwards pass using rank-dAD for communication.
            This method rank-reduces the batch dimension for linear networks (or batch*sequence length
            for recurrent networks). 
           Communication and computation should be efficient when the rank and number of power iterations
            are relatively small.
        """
        hook = self.model.hook
        start = time.time()
        comm = 0
        for key, parameters in zip(hook.keys, hook.parameters):
            input_activations = hook.forward_stats[key]['input'][0]
            deltas = hook.backward_stats[key]['output'][0]
            act, delta = mm_flatten(input_activations, deltas)

            """ Rank reduce with PowerIteration """
            delta_local_reduced, act_local_reduced = power_iteration_BC(
                delta.T,
                act.T,                
                device=act.device,
                rank=pi_effective_rank,
                numiterations=pi_numiterations,
                use_sigma=use_sigma
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
            for x in act_gathered:
                comm += x.element_size() * x.nelement()
            for x in delta_gathered:
                comm += x.element_size() * x.nelement()

            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)
            #print("act_gathered.shape ", act_gathered.shape)
            #print("delta_gathered.shape ", delta_gathered.shape)
            delta_global_reduced, act_global_reduced = power_iteration_BC(
                delta_gathered.T,
                act_gathered.T,                
                device=act_gathered.device,
                rank=pi_effective_rank,
                numiterations=pi_numiterations,
                use_sigma=pi_use_sigma
            )
            comm += delta_global_reduced.element_size() * delta_global_reduced.nelement()
            comm += act_global_reduced.element_size() * act_global_reduced.nelement()

            #parameters['weight'].grad.data = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            parameters['weight'].grad.data = (act_global_reduced.mm(delta_global_reduced.T)).T.contiguous()
            if "bias" in parameters.keys():
                parameters["bias"].grad.data = delta_gathered.sum(0)
        print("Done with batch...")
        print(time.time() - start)
        return comm


    
    def _rankdad_p2p_backward(self, effrank=3, numiterations=1, **kwargs):
        """This function performs the backwards pass using rank-dAD for communication.
            This method rank-reduces the batch dimension for linear networks (or batch*sequence length
            for recurrent networks). 
           Communication and computation should be efficient when the rank and number of power iterations
            are relatively small.
        """
        hook = self.model.hook
        comm = 0
        start = time.time()
        for key, parameters in zip(hook.keys, hook.parameters):
            input_activations = hook.forward_stats[key]['input'][0]
            deltas = hook.backward_stats[key]['output'][0]
            act, delta = mm_flatten(input_activations, deltas)            
            """ Rank reduce with PowerIteration """
            # print("delta.shape", delta.shape)
            # print("act.shape", act.shape)
            delta_local_reduced, act_local_reduced = power_iteration_BC(
                delta.T,
                act.T,                
                device=act.device,
                rank=effrank,
                numiterations=numiterations
            )
            """Rank-reduction end. """

            # print("Finished with reduction")
            # first send act_local
            if torch.distributed.get_rank() == 0:
                act_send = None
                delta_send = None
            else:
                act_send = act_local_reduced
                delta_send = delta_local_reduced
            act_gathered = point_to_master(act_send, torch.distributed.get_world_size(), act.device)
            delta_gathered = point_to_master(delta_send, torch.distributed.get_world_size(), act.device)
            # print("Finished Communication")
            # print("act_gathered ", len(act_gathered), [a.size() for a in act_gathered])
            # print("delta_gathered ", len(delta_gathered), [a.size() for a in delta_gathered])

            for x in act_gathered:
                comm += x.element_size() * x.nelement()
            for x in delta_gathered:
                comm += x.element_size() * x.nelement()
        
            delta_global_reduced = None
            act_global_reduced = None
            if torch.distributed.get_rank() == 0:
                act_gathered.insert(0, act_local_reduced)
                delta_gathered.insert(0, delta_local_reduced)
                act_full = torch.cat(act_gathered, 1).t()
                delta_full = torch.cat(delta_gathered, 1).t()
                #print("act_full.shape ", act_full.shape)
                #print("delta_full.shape ", delta_full.shape)
                delta_global_reduced, act_global_reduced = power_iteration_BC(
                    delta_full.T,
                    act_full.T,                
                    device=act.device,
                    rank=effrank,
                    numiterations=numiterations
                )
            delta_global_reduced = coordinated_broadcast(delta_global_reduced, self.device, 0)
            act_global_reduced = coordinated_broadcast(act_global_reduced, self.device, 0)
            # print("act_global_reduced shape", act_global_reduced.shape)
            # print("delta_global_reduced shape", delta_global_reduced.shape)            
            comm += delta_global_reduced.element_size() * delta_global_reduced.nelement()
            comm += act_global_reduced.element_size() * act_global_reduced.nelement()

            #parameters['weight'].grad.data = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            #print("TEST")
            #test = (act_global_reduced.mm(delta_global_reduced.T)).T.contiguous()
            #print(parameters['weight'].grad.shape, test.shape)
            #test = delta_global_reduced.sum(0)
            #test2 = delta_global_reduced.sum(1)
            #print(parameters["bias"].grad.shape, test.shape, test2.shape)

            parameters['weight'].grad.data = (act_global_reduced.mm(delta_global_reduced.T)).T.contiguous()
            if "bias" in parameters.keys():
                parameters["bias"].grad.data = delta_global_reduced.sum(1)
        print("Done with batch...")
        print(time.time() - start, " seconds")
        return comm

    def _topk_backward(self, k=3, **kwargs):
        """This function performs the backwards pass using rank-dAD for communication.
            This method rank-reduces the batch dimension for linear networks (or batch*sequence length
            for recurrent networks). 
           Communication and computation should be efficient when the rank and number of power iterations
            are relatively small.
        """
        hook = self.model.hook
        comm = 0
        for key, parameters in zip(hook.keys, hook.parameters):
            input_activations = hook.forward_stats[key]['input'][0]
            deltas = hook.backward_stats[key]['output'][0]
            act, delta = mm_flatten(input_activations, deltas)

            """ Rank reduce with PowerIteration """
            delta_local_reduced = delta.T[:, :k]
            act_local_reduced = act.T[:, :k]
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
            for x in act_gathered:
                comm += x.element_size() * x.nelement()
            for x in delta_gathered:
                comm += x.element_size() * x.nelement()

            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)

            parameters['weight'].grad.data = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            if "bias" in parameters.keys():
                parameters["bias"].grad.data = delta_gathered.sum(0)
        return comm

    def _dsgd_backward(self, **kwargs):
        """This method replicates distributed SGD with all-gather. 
        We use this as a baseline rather than the pytorch native method as a fair 
        comparison, since the native method is implemented in highly optimized C.
        """
        size = torch.distributed.get_world_size()
        comm = 0
        for param in self.model.parameters():
            grad_gathered = [torch.zeros_like(param.grad.data) for _ in range(size)]
            torch.distributed.all_gather(grad_gathered, param.grad.data)           
            for x in grad_gathered:
                comm += x.element_size() * x.nelement()
            param.grad.data = torch.stack(grad_gathered).sum(0) / float(size)  
        return comm  

    def on_loader_end(self, runner):
        """This function resolves metrics at the end of runtime"""
        for key in ["loss", "cumulative_runtime"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

if __name__=="__main__":
    print("ok")
    kwargs = {"num_nodes": 4, "rank": 4}
    dr = DistributedRunner(**kwargs)