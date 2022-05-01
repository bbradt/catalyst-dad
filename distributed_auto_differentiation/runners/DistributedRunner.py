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
                matrix_approximation_rank=1, start_powerSGD_iter=10, use_error_feedback=True, 
                warm_start=True, **kwargs):
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
            for key in ["loss", "bits", "runtime"]
        }
    def handle_batch(self, batch): 
        """This function handles the batch computation, and is where 
                the appropriate distributed function is selected for sharing statistics. 
                Runtime and other metrics are also evaluated here.
        """
        input, target = batch
        print("DR(81): Before forward call")
        logits = self.model(input)
        if type(logits) is tuple:
            logits = logits[0]
        #print("DR(83): ", logits.shape, target.shape)
        #print(logits, target.flatten())
        loss = self.criterion(logits, target.flatten())
        if not os.path.exists('/data/users2/bbaker/projects/dist_autodiff/model_graph.png'):
            make_dot(loss, params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True).render('/data/users2/bbaker/projects/dist_autodiff/model_graph', 'png')
        self.batch_metrics.update({"loss": loss})     
        self.batch_metrics.update({"bits": 0})        
        #print(list(self.meters.keys()))
        
        # Training Only
        if self.is_train_loader:
            self.optimizer.zero_grad()
            dprint("Before backward call")
            loss.backward(retain_graph=True)
            start_time = time.time()
            #print("WHATS MODE ", self.mode)
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
            elif self.mode == "powersgd":
                comm = self._powersgd_backward(**self.kwargs)
            # update batch runtime
            runtime = time.time() - start_time
            self.batch_metrics.update({"runtime": runtime})
            #self.meters["cumulative_runtime"].update(self.batch_metrics["runtime"], 1)
            self.batch_metrics.update({"bits": comm})
            for key in ["loss", "bits", "runtime"]:
                try:
                    self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
                except Exception:
                    self.meters[key].update(self.batch_metrics[key], self.batch_size)
            # Step optimizer
            self.optimizer.step()  
        else:
            for key in ["loss"]:
                self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
        self.model.hook.clear()
        logits = logits.argmax(1).reshape(target.shape)
        #print(target.shape, logits.shape)
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
            dprint(key)
            #dprint(type(input_activations[0]), type(deltas[0]))
            ts = len(input_activations)
            #dprint(len(input_activations), len(deltas))
            test2 = []
            if len(input_activations) == 1:
                input_activations = input_activations[0]
                deltas = deltas[0]                
                test = []
                """
                elif len(input_activations) == 32:
                    test = []
                    device = input_activations[0].device
                    input_activations_2 = [torch.zeros((32, input_activations[0].shape[1])).to(device) for i in range(98)]
                    deltas_2 = [torch.zeros((32, deltas[0].shape[1])).to(device) for i in range(98)]
                    for t in range(98):
                        for i in range(32):
                            input_activations_2[t][i, ...] = input_activations[i][t, ...]
                            deltas_2[t][i, ...] = deltas[i][t, ...]
                    #dprint([t.shape for t in input_activations_2])
                    #dprint([t.shape for t in deltas_2])
                    test = [(t1.T @ t2).T for t1, t2 in zip(input_activations_2, deltas_2) if len(t1.shape) > 1]
                    input_activations = torch.concat([t for t in input_activations_2 if len(t.shape) > 1], 0)
                    deltas = torch.concat([t for t in deltas_2 if len(t.shape) > 1], 0)
                """
            else:
                #print([t.shape for t in input_activations])
                #print([t.shape for t in deltas])
                test = [(t1.T @ t2).T for t1, t2 in zip(input_activations, deltas) if len(t1.shape) > 1]
                test2 = []
                for act2, delt2 in zip(input_activations, deltas):
                    gg = torch.zeros_like(test[0]).to(test[0].device)
                    for i in range(act2.shape[0]):
                        a = act2[i, ...].view(1, act2.shape[1])
                        b = delt2[i, ...].view(1, delt2.shape[1])
                        gg += (a.T @ b).T
                    test2.append(gg)
                #for t1, t2 in zip(input_activations, deltas)
                input_activations = torch.concat([t for t in input_activations if len(t.shape) > 1], 0).double()
                deltas = torch.concat([t for t in deltas if len(t.shape) > 1], 0).double()                                 
                #dprint([len(t1) for t1 in input_activations])
                #dprint(len(test))
                #test = []
                #input_activations = input_activations[-1]
                #deltas = deltas[-1]
            # Flatten matrices prior to computation
            dprint(input_activations.shape, deltas.shape)
            act, delta = mm_flatten(input_activations, deltas)
            #dprint(act.shape, delta.shape)
            act_gathered = [torch.zeros_like(act).to(act.device).contiguous() for _ in range(torch.distributed.get_world_size())]
            delta_gathered = [torch.zeros_like(delta).to(delta.device).contiguous() for _ in range(torch.distributed.get_world_size())]
            # Communicate Forward and Backward Stats
            torch.distributed.all_gather(act_gathered, act.contiguous())
            torch.distributed.all_gather(delta_gathered, delta.contiguous())
            for x in act_gathered:
                comm += x.element_size() * x.nelement()
            for x in delta_gathered:
                comm += x.element_size() * x.nelement()
            # Concatenate
            act_gathered = torch.cat(act_gathered)
            delta_gathered = torch.cat(delta_gathered)

            # Product computation of gradient            
            dprint(act_gathered.shape, delta_gathered.shape, module.weight.grad.shape)
            new_grad = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            #new_grad /= act_gathered.shape[0]
            
            dprint(torch.norm(module.weight.grad - new_grad))

            if len(test) > 0:
                dprint(torch.norm(module.weight.grad - torch.stack(test, -1).sum(-1))) 
            if len(test2) > 0:
                dprint(torch.norm(module.weight.grad - torch.stack(test2, -1).sum(-1))) 
            hook.forward_stats[key] = None
            hook.backward_stats[key] = None
            module.weight.grad.data = new_grad.data
            dprint(torch.norm(module.weight.grad - new_grad))
            #if "bias" in parameters.keys():
            #    parameters["bias"].grad.data = delta_gathered.sum(0)
            input()
        return comm

    def _powersgd_backward(self, rank=2, **kwargs):
        #print("HEYYYYY")
        hook = self.model.hook
        start = time.time()
        comm = 0
        for key, parameters in zip(hook.keys, hook.parameters):
            grad = parameters['weight'].grad.clone()
            if key not in self.state.Ps.keys():
                self.state.Ps[key] = None
                self.state.Qs[key] = None
                Q = torch.randn(grad.shape[1], rank).to(grad.device)
            else:
                Q = self.state.Qs[key]
            P = grad @ Q
            P = gram_schmidt(P)
            Q = grad.t() @ P                    
            self.state.Qs[key] = Q.clone()
            self.state.Ps[key] = P.clone()    
            torch.distributed.all_reduce(P)
            torch.distributed.all_reduce(Q)
            comm += P.element_size() * P.nelement()
            comm += Q.element_size() * Q.nelement()
            grad = P @ Q.t()
            parameters['weight'].grad.data = grad.clone()
        #print("Done with batch...")
        #print(time.time() - start)
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
            input_activations = hook.forward_stats[key]['input']
            grad = module.weight.grad.clone()
            deltas = hook.backward_stats[key]['output']
            #print(input_activations.shape, )
            #print(key)
            #print(len(input_activations), len(deltas), grad.shape)
            #input()
            if len(input_activations) == 1:
                input_activations = input_activations[0]
                deltas = deltas[0]                
            else:
                #print([t[0].shape for t in input_activations])
                #print([t[0].shape for t in deltas])
                input_activations = torch.concat([t[0] for t in input_activations if len(t[0].shape) > 1], 0).double()
                deltas = torch.concat([t[0] for t in deltas if len(t[0].shape) > 1], 0).double()
                #input_activations = (input_activations, )
                #deltas = (deltas, )                
            #print(input_activations.shape, deltas.shape)
            #input()
            act, delta = mm_flatten(input_activations, deltas)
            #print(act.shape, delta.shape, grad.shape)
            full_recon = delta.t() @ act
            """ Rank reduce with PowerIteration """
            rank = min(delta.shape[0], act.shape[0])
            delta_local_reduced, act_local_reduced = power_iteration_BC(
                delta.T,
                act.T,                
                device=act.device,
                rank=rank,
                numiterations=pi_numiterations,
                #use_sigma=True
            )
            #delta_local_reduced = delta.T.contiguous()
            #act_local_reduced = act.T.contiguous()
            #print(delta_local_reduced.shape, act_local_reduced.shape, grad.shape)
            partial_recon = delta_local_reduced @ act_local_reduced.t()
            #print(torch.norm(partial_recon - grad))
            """Rank-reduction end. """

            """ Pick Max rank of the world and pad to match """
            max_rank = torch.Tensor([delta_local_reduced.shape[1]]).to(act.device)
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
                rank=rank,
                numiterations=pi_numiterations,
                #use_sigma=pi_use_sigma
            )
            #delta_global_reduced = delta_gathered.T.contu
            #act_global_reduced = act_gathered.T.contiguous()
            comm += delta_global_reduced.element_size() * delta_global_reduced.nelement()
            comm += act_global_reduced.element_size() * act_global_reduced.nelement()

            #parameters['weight'].grad.data = (act_gathered.T.mm(delta_gathered)).T.contiguous()
            module.weight.grad.data = (act_global_reduced.mm(delta_global_reduced.T)).T.contiguous()
            #if "bias" in parameters.keys():
            #    parameters["bias"].grad.data = delta_gathered.sum(0)
        #print("Done with batch...")
        #print(time.time() - start)
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
            delta_global_reduced = coordinated_broadcast(delta_global_reduced, act.device, 0)
            act_global_reduced = coordinated_broadcast(act_global_reduced, act.device, 0)
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
        #print("Done with batch...")
        #print(time.time() - start, " seconds")
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
        for key in ["loss"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

if __name__=="__main__":
    print("ok")
    kwargs = {"num_nodes": 4, "rank": 4}
    dr = DistributedRunner(**kwargs)