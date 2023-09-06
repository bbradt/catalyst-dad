from catalyst import dl, metrics
import torch
import torch.nn.functional as F
import time
import os
import inspect
from torchviz import make_dot
from distributed_auto_differentiation.utils import mm_flatten, power_iteration_BC, point_to_master, coordinated_broadcast, dprint

def invalidArgs(func, argdict):
    args, varargs, varkw, defaults = inspect.getargspec(func)
    if varkw: return set()  # All accepted
    return set(argdict) - set(varkw)

class RunnerState:
    def __init__(self):
        self.error_dict = dict()
        self.Ps = dict()
        self.Qs = dict()
        self.iter = 0

def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(1)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[:, k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu

class DistributedRunner(dl.Runner):
    """This is the distributed Runner for Catalyst, which handles batch activity
            This class also contains the important functions needed for dSGD, dAD, and rank-dAD.
    """
    def __init__(self, model=None, criterion=None, optimizer=None, distributed_mode="dsgd", rank=None,
                 **kwargs):
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
        #print("KWARGS", kwargs)
        #exit()
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion
        self.kwargs = kwargs
        self.mode = distributed_mode.lower()
        self.rank = rank        
        self.state = RunnerState()

    def on_loader_start(self, runner):
        """This function initializes measured metrics on dataloader start"""
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveMetric(compute_on_call=False)
            for key in ["loss", "bits", "runtime", "graderr", "comm_time", "comp_time"]
        }

    def handle_batch(self, batch): 
        """This function handles the batch computation, and is where 
                the appropriate distributed function is selected for sharing statistics. 
                Runtime and other metrics are also evaluated here.
        """
        input, target = batch
        logits = self.model(input)
        if type(logits) is tuple:
            logits = logits[0]
        loss = self.criterion(logits, target.flatten())
        self.batch_metrics.update({"loss": loss})     
        self.batch_metrics.update({"bits": 0}) 
        self.batch_metrics.update({"runtime": 0})   
        self.batch_metrics.update({"graderr": 0})     
        self.batch_metrics.update({"comp_time": 0})
        self.batch_metrics.update({"comm_time": 0})
        
        # Training Only
        if self.is_train_loader:
            self.optimizer.zero_grad()
            
            #loss.backward(retain_graph=True)            
            loss.backward()
            start_time = time.time()        
            # Below is the switch for different distributed AD methods
            if self.mode == "dad":
                comm, comm_time, comp_time, graderr = self._dad_backward(**self.kwargs)
            elif self.mode == "dsgd":
                comm, comm_time, comp_time, graderr = self._dsgd_backward(**self.kwargs)
            elif self.mode == "rankdad":
                comm, comm_time, comp_time, graderr = self._rankdad_allgather_backward(**self.kwargs)
            elif self.mode == "rankdad_oneway":
                comm, comm_time, comp_time, graderr = self._rankdad_og_backward(**self.kwargs)
            elif self.mode == "topk":
                comm, comm_time, comp_time, graderr = self._topk_backward(**self.kwargs)
            elif self.mode == "powersgd":
                comm, comm_time, comp_time, graderr = self._powersgd_backward(**self.kwargs)
            else:
                comm = 0
            # update batch runtime
            runtime = time.time() - start_time
            self.batch_metrics.update({"runtime": runtime})
            self.batch_metrics.update({"bits": comm})
            self.batch_metrics.update({"graderr": graderr})
            self.batch_metrics.update({"comp_time": comp_time})
            self.batch_metrics.update({"comm_time": comm_time})
            for key in ["loss", "bits", "runtime", "graderr", "comp_time", "comm_time"]:
                try:
                    if key == 'loss':
                        self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
                    else:
                        self.meters[key].update(self.batch_metrics[key].item(), 1)
                except Exception as e:
                    if key == 'loss':
                        self.meters[key].update(self.batch_metrics[key], self.batch_size)
                    else:
                        self.meters[key].update(self.batch_metrics[key], 1)
            # Step optimizer
            self.optimizer.step()  
            self.model.hook.clear()
        else:
            for key in ["loss"]:
                self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)                
            for key in ["bits", "runtime", "graderr", "comp_time", "comm_time"]:
                self.meters[key].update(self.batch_metrics[key], 1)
        self.model.hook.clear()
        preds = logits.argmax(1).reshape(target.shape)
        self.batch = dict(features=input, targets=target, logits=logits, preds=preds)        


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
        comptime = 0
        commtime = 0
        graderr= 0

        modules = list(
            [
                module
                for i, module in enumerate(self.model.modules())
                if module.__class__.__name__ in ["Linear"]
            ]
        )
        rev_mod = modules[::-1]
        self.orders = dict()        
        # Iterate through the modules in reverse-order
        for m_i, module in enumerate(rev_mod):            
            key = str(module) + str(m_i)
            # Grab forward and backward stats from hook
            input_activations = hook.forward_stats[key]['input']
            deltas = hook.backward_stats[key]['output']
            ##dprint(key)
            ##dprint(type(input_activations[0]), type(deltas[0]))
            ts = len(input_activations)
            ##dprint(len(input_activations), len(deltas))
            test2 = []
            if len(input_activations) == 1:
                input_activations = input_activations[0]
                if type(input_activations) is tuple:
                    input_activations = input_activations[0]
                deltas = deltas[0]                
            else:
                deltas.reverse()                
                input_activations = torch.concat([t for t in input_activations if len(t.shape) > 1], 0)
                deltas = torch.concat([t for t in deltas if len(t.shape) > 1], 0)                                 
            # Flatten matrices prior to computation
            act, delta = mm_flatten(input_activations, deltas)
            act_gathered = [torch.zeros_like(act).to(act.device).contiguous() for _ in range(torch.distributed.get_world_size())]
            delta_gathered = [torch.zeros_like(delta).to(delta.device).contiguous() for _ in range(torch.distributed.get_world_size())]
            # Communicate Forward and Backward Stats
            comm_start_time = time.time()
            torch.distributed.all_gather(act_gathered, act.contiguous())
            torch.distributed.all_gather(delta_gathered, delta.contiguous())
            commtime += time.time() - comm_start_time
            for x in act_gathered:
                comm += x.element_size() * x.nelement()
            for x in delta_gathered:
                comm += x.element_size() * x.nelement()
            # Concatenate
            comp_start_time = time.time()
            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)

            # Product computation of gradient                        
            new_grad = (act_gathered.T.mm(delta_gathered)).T.contiguous()            
            comptime += time.time() - comp_start_time
            if hasattr(module, 'weight'):
                graderr+=  torch.linalg.norm(module.weight.grad - new_grad).item()
            hook.forward_stats[key] = None
            hook.backward_stats[key] = None
            module.weight.grad.data = new_grad.data
        return comm, commtime, comptime, graderr

    def _powersgd_backward(self, psgd_rank=2, **kwargs):
        hook = self.model.hook
        comm = 0
        comptime = 0
        commtime = 0
        graderr= 0
        for key, parameters in zip(hook.keys, hook.parameters):
            grad = parameters['weight'].grad.clone()
            if key not in self.state.Qs.keys():
                Q = torch.randn(parameters['weight'].shape[-1], psgd_rank).to(grad.device)
            else:
                Q = self.state.Qs[key]
            P = grad @ Q
            P = gram_schmidt(P)
            Q = grad.t() @ P                    
            self.state.Qs[key] = Q.clone()
            self.state.Ps[key] = P.clone()    
            #Pcopy = P.clone()
            #Precv = torch.zeros_like(P).to(P.device)
            #Qrecv = torch.zeros_like(Q).to(Q.device)
            Precv = [torch.zeros_like(P).to(P.device) for _ in range(torch.distributed.get_world_size())]
            Qrecv = [torch.zeros_like(Q).to(Q.device) for _ in range(torch.distributed.get_world_size())]
            #torch.distributed.all_reduce(P)
            #torch.distributed.all_reduce(Q)
            compstart = time.time()
            torch.distributed.all_gather(Precv, P.contiguous())            
            torch.distributed.all_gather(Qrecv, Q.contiguous())            
            commtime += time.time() - compstart
            #comm += P.element_size() * P.nelement()
            #comm += Q.element_size() * Q.nelement()
            for x in Precv:
                comm += x.element_size() * x.nelement()
            for x in Qrecv:
                comm += x.element_size() * x.nelement()
            compstart = time.time()
            P = torch.mean(torch.stack(Precv, -1), -1)
            Q = torch.mean(torch.stack(Qrecv, -1), -1)
            grad = P @ Q.t()
            comptime += time.time() - compstart
            graderr+=  torch.linalg.norm(parameters['weight'].grad - grad).item()

            ##dprint("Diff ", torch.norm(parameters['weight'].grad - grad))            
            parameters['weight'].grad.data = grad.data
            ##dprint("Diff ", torch.norm(parameters['weight'].grad - grad))
            #input()
            
        #print("Done with batch...")
        #print(time.time() - start)
        return comm, commtime, comptime, graderr

    def _rankdad_allgather_backward(self, pi_effective_rank=2, pi_numiterations=1, pi_tolerance=0.001, pi_use_sigma=False, **kwargs):
        """This function performs the backwards pass using rank-dAD for communication.
            This method rank-reduces the batch dimension for linear networks (or batch*sequence length
            for recurrent networks). 
           Communication and computation should be efficient when the rank and number of power iterations
            are relatively small.
        """
        hook = self.model.hook
        start = time.time()
        comm = 0
        comptime = 0
        commtime = 0
        #return comm
        modules = list(
            [
                module
                for i, module in enumerate(self.model.modules())
                if module.__class__.__name__ in ["Linear"]
            ]
        )
        rev_mod = modules[::-1]
        #print("first backward")
        for m_i, module in enumerate(rev_mod):
            key = str(module) + str(m_i)
            #dprint(key)
            input_activations = hook.forward_stats[key]['input']
            deltas = hook.backward_stats[key]['output']
            
            if len(input_activations) == 1:
                input_activations = input_activations[0]
                if type(input_activations) is tuple:
                    input_activations = input_activations[0]
                deltas = deltas[0]      
            else:
                deltas.reverse()
                input_activations = torch.concat([t for t in input_activations if len(t.shape) > 1], 0)
                deltas = torch.concat([t for t in deltas if len(t.shape) > 1], 0)
            act, delta = mm_flatten(input_activations, deltas)            
            """ Rank reduce with PowerIteration """      
            #dprint(act.shape, delta.shape)        
            compstart = time.time()          
            delta_local_reduced, act_local_reduced = power_iteration_BC(
                delta.T,
                act.T,                
                device=act.device,
                rank=pi_effective_rank,
                numiterations=pi_numiterations,
                tol=pi_tolerance
                #use_sigma=True
            )
            comptime += time.time() - compstart
            #correct_grad = module.weight.grad
            ##dprint(torch.linalg.matrix_rank(correct_grad))
            #grad_partial_reconstruct = (act_local_reduced @ delta_local_reduced.T).T
            #dprint(torch.norm(correct_grad - grad_partial_reconstruct))
            """Rank-reduction end. """
            ##dprint(delta_local_reduced.shape, act_local_reduced.shape)

            """ Pick Max rank of the world and pad to match """
            commstart = time.time()
            max_rank = torch.Tensor([delta_local_reduced.shape[1]]).to(act.device)
            torch.distributed.all_reduce(max_rank, torch.distributed.ReduceOp.MAX)
            commtime += time.time() - commstart

            if max_rank > delta_local_reduced.shape[1]:
                _pad = (0, int(max_rank.item() - delta_local_reduced.shape[1]))
                act_local_reduced = F.pad(act_local_reduced, _pad)
                delta_local_reduced = F.pad(delta_local_reduced, _pad)
            act_gathered = [torch.zeros_like(act_local_reduced.T) for _ in range(torch.distributed.get_world_size())]
            delta_gathered = [torch.zeros_like(delta_local_reduced.T) for _ in range(torch.distributed.get_world_size())]
            #dprint(act_local_reduced.shape, delta_local_reduced.shape)
            commstart = time.time()
            torch.distributed.all_gather(act_gathered, act_local_reduced.T)
            torch.distributed.all_gather(delta_gathered, delta_local_reduced.T)
            commtime += time.time() - commstart
            for x in act_gathered:
                comm += x.element_size() * x.nelement()
            for x in delta_gathered:
                comm += x.element_size() * x.nelement()     
            compstart = time.time()       
            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)
            #dprint(act_gathered.shape, delta_gathered.shape, correct_grad.shape)
            #grad_before_global = (act_gathered.T @ delta_gathered).T
            #dprint(torch.norm(correct_grad - grad_before_global))            
            delta_global_reduced, act_global_reduced = power_iteration_BC(
                delta_gathered.T,
                act_gathered.T,                
                device=act_gathered.device,
                rank=pi_effective_rank,
                numiterations=pi_numiterations,
                tol=pi_tolerance
                #use_sigma=pi_use_sigma
            )
            comptime += time.time() - compstart
            #grad_after_global = (act_global_reduced @ delta_global_reduced.T).T
            #dprint(torch.norm(correct_grad - grad_after_global))
            comm += delta_global_reduced.element_size() * delta_global_reduced.nelement()
            comm += act_global_reduced.element_size() * act_global_reduced.nelement()
            compstart = time.time()
            new_grad = (act_global_reduced.mm(delta_global_reduced.T)).T.contiguous()
            comptime += time.time() - compstart
            graderr += torch.norm(module.weight.grad - new_grad)
            module.weight.grad.data = new_grad.data
            #input()
        return comm, commtime, comptime, graderr    

    def _rankdad_og_backward(self, pi_effective_rank=2, pi_numiterations=1, pi_tolerance=0.001, pi_use_sigma=False, **kwargs):
        """This function performs the backwards pass using rank-dAD for communication.
            This method rank-reduces the batch dimension for linear networks (or batch*sequence length
            for recurrent networks). 
           Communication and computation should be efficient when the rank and number of power iterations
            are relatively small.
        """
        hook = self.model.hook
        start = time.time()
        comm = 0
        commtime = 0
        comptime = 0
        graderr = 0
        #return comm
        modules = list(
            [
                module
                for i, module in enumerate(self.model.modules())
                if module.__class__.__name__ in ["Linear"]
            ]
        )
        rev_mod = modules[::-1]
        #print("first backward")
        for m_i, module in enumerate(rev_mod):
            key = str(module) + str(m_i)
            #dprint(key)
            input_activations = hook.forward_stats[key]['input']
            deltas = hook.backward_stats[key]['output']            
            #dprint("Module %s" % key)
            #dprint("Input activations shape len/type %s/%s and delta len/type %s/%s" % (len(input_activations), type(input_activations), len(deltas), type(deltas)))                
            if len(input_activations) == 1:
                input_activations = input_activations[0]
                if type(input_activations) is tuple:
                    input_activations = input_activations[0]
                deltas = deltas[0]                
            else:
                deltas.reverse()
                input_activations = torch.concat([t for t in input_activations if len(t.shape) > 1], 0)
                deltas = torch.concat([t for t in deltas if len(t.shape) > 1], 0)
            #dprint("Input activations shape len/type %s/%s and delta len/type %s/%s" % (len(input_activations), type(input_activations), len(deltas), type(deltas)))                
            #dprint("Input activations[0] shape len/type %s/%s and delta len/type %s/%s" % (len(input_activations[0]), type(input_activations[0]), len(deltas), type(deltas)))                
            act, delta = mm_flatten(input_activations, deltas)            
            """ Rank reduce with PowerIteration """      
            #dprint(act.shape, delta.shape)      
            compstart = time.time()            
            delta_local_reduced, act_local_reduced = power_iteration_BC(
                delta.T,
                act.T,                
                device=act.device,
                rank=pi_effective_rank,
                numiterations=pi_numiterations,
                tol=pi_tolerance
                #use_sigma=True
            )
            comptime += time.time() - compstart
            
            #correct_grad = module.weight.grad
            ##dprint(torch.linalg.matrix_rank(correct_grad))
            #grad_partial_reconstruct = (act_local_reduced @ delta_local_reduced.T).T
            #dprint(torch.norm(correct_grad - grad_partial_reconstruct))
            """Rank-reduction end. """
            ##dprint(delta_local_reduced.shape, act_local_reduced.shape)

            """ Pick Max rank of the world and pad to match """
            commstart = time.time()
            max_rank = torch.Tensor([delta_local_reduced.shape[1]]).to(act.device)
            torch.distributed.all_reduce(max_rank, torch.distributed.ReduceOp.MAX)
            commtime += time.time() - commstart
            if max_rank > delta_local_reduced.shape[1]:
                _pad = (0, int(max_rank.item() - delta_local_reduced.shape[1]))
                act_local_reduced = F.pad(act_local_reduced, _pad)
                delta_local_reduced = F.pad(delta_local_reduced, _pad)
            act_gathered = [torch.zeros_like(act_local_reduced.T) for _ in range(torch.distributed.get_world_size())]
            delta_gathered = [torch.zeros_like(delta_local_reduced.T) for _ in range(torch.distributed.get_world_size())]
            #dprint(act_local_reduced.shape, delta_local_reduced.shape)
            commstart = time.time()
            torch.distributed.all_gather(act_gathered, act_local_reduced.T)
            torch.distributed.all_gather(delta_gathered, delta_local_reduced.T)
            commtime += time.time() - commstart
            for x in act_gathered:
                comm += x.element_size() * x.nelement()
            for x in delta_gathered:
                comm += x.element_size() * x.nelement()
            compstart = time.time()            
            act_global_reduced = torch.cat(act_gathered).T
            delta_global_reduced = torch.cat(delta_gathered)
            #dprint(act_gathered.shape, delta_gathered.shape, correct_grad.shape)
            #grad_before_global = (act_gathered.T @ delta_gathered).T
            #dprint(torch.norm(correct_grad - grad_before_global))                        
            new_grad = (act_global_reduced.mm(delta_global_reduced)).T.contiguous()
            comptime += time.time() - compstart
            graderr += torch.norm(module.weight.grad - new_grad)
            module.weight.grad.data = new_grad.data
            #input()
        return comm, commtime, comptime, graderr    

    def _topk_backward(self, k=3, **kwargs):
        """This function performs the backwards pass using rank-dAD for communication.
            This method rank-reduces the batch dimension for linear networks (or batch*sequence length
            for recurrent networks). 
           Communication and computation should be efficient when the rank and number of power iterations
            are relatively small.
        """
        hook = self.model.hook
        comm = 0
        commtime = 0
        comptime = 0
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
            commstart = time.time()
            torch.distributed.all_reduce(max_rank, torch.distributed.ReduceOp.MAX)
            commtime += time.time() - commstart

            if max_rank > delta_local_reduced.shape[1]:
                _pad = (0, int(max_rank.item() - delta_local_reduced.shape[1]))
                act_local_reduced = F.pad(act_local_reduced, _pad)
                delta_local_reduced = F.pad(delta_local_reduced, _pad)
            act_gathered = [torch.zeros_like(act_local_reduced.T) for _ in range(torch.distributed.get_world_size())]
            delta_gathered = [torch.zeros_like(delta_local_reduced.T) for _ in range(torch.distributed.get_world_size())]
            commstart = time.time()
            torch.distributed.all_gather(act_gathered, act_local_reduced.T)
            torch.distributed.all_gather(delta_gathered, delta_local_reduced.T)
            commtime += time.time() - commstart
            for x in act_gathered:
                comm += x.element_size() * x.nelement()
            for x in delta_gathered:
                comm += x.element_size() * x.nelement()
            compstart = time.time()
            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)
            newgrad = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            comptime += time.time() - compstart
            graderr = torch.norm(newgrad - parameters['weight'].grad).item()
            parameters['weight'].grad.data = newgrad
            if "bias" in parameters.keys():
                parameters["bias"].grad.data = delta_gathered.sum(0)
            comptime += time.time() - compstart
        return comm, commtime, comptime, graderr

    def _dsgd_backward(self, **kwargs):
        """This method replicates distributed SGD with all-gather. 
        We use this as a baseline rather than the pytorch native method as a fair 
        comparison, since the native method is implemented in highly optimized C.
        """
        size = torch.distributed.get_world_size()
        comm = 0
        commtime = 0
        comptime = 0
        for param in self.model.parameters():
            grad_gathered = [torch.zeros_like(param.grad.data) for _ in range(size)]
            commstart = time.time()
            torch.distributed.all_gather(grad_gathered, param.grad.data)           
            commtime += time.time() - commstart
            for x in grad_gathered:
                comm += x.element_size() * x.nelement()
            compstart = time.time()
            newgrad = torch.stack(grad_gathered).sum(0) / float(size)  
            comptime += time.time() - compstart
            graderr = torch.norm(param.grad - newgrad).item()
            param.grad.data = newgrad
        return comm, commtime, comptime, graderr

    def on_loader_end(self, runner):
        """This function resolves metrics at the end of runtime"""
        for key in ["loss", "bits", "runtime", "graderr"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

if __name__=="__main__":
    print("ok")
    kwargs = {"num_nodes": 4, "rank": 4}
    dr = DistributedRunner(**kwargs)