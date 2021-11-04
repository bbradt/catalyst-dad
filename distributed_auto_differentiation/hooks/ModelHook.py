"""
ModelHook.py

This module implements a class which uses Pytorch forward and backward hooks
    to collect statistics during the forward and backward passes of the algorithm.
    The class allows users to attach these hooks directly to pytorch modules so that
    they will automatically gather statistics which can then be used to update gradients.
"""

import torch.nn as nn

try:
    import pydevd
except Exception:
    pass


class ModelHook:
    def __init__(
        self, model, verbose=False, layer_names=None, register_self=True, save=True
    ):        
        """A class to wrap and apply Pytorch's forward and backward hooks to gather statistics
            during forward and backward pass.

           Args:
            model: torch.nn.Module - a pytorch module or model onto which hooks will be applied.
           Kwargs:
            layer_names: list<str> - a list of strings indicating the kinds of layers to include.
                E.g. if a model with both convolutional and linear layers is included, and 
                ["Linear"] is supplied, only Linear layers will be treated as distributed, and 
                other layers will be local.
            register_self: bool - boolean switch if the hook is initially off (expecting to be manually registered elsewhere)
            save: bool - boolean switch to save statistics during training (this is mostly used for debugging purposes on memory-limited machines)        
        """
        self.model = model
        self.verbose = verbose
        self.layer_names = layer_names
        self.keys = []
        self.parameters = []
        self.orders = dict()
        if register_self:
            self.register_hooks(self.model, layer_names)
        self.save = save
        self.clear()

    def clear(self):
        """Clear all saved statistics"""
        self.grads = dict()
        self.forward_stats = dict()
        self.backward_stats = dict()
        self.backward_return = None
        self.batch_indices = None

    def forward_hook_fn(self, module, input, output):
        """This is a generic forward hook function which saves all statistics from the forward pass
                during training. This function can be overwritten by a subclass, and the subclass' function
                should be automatically registered in place of this one. 

            This function follows the structure of forward hooks as explained in the pytorch
                docs: https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html

            Args:
                module - torch.nn.Module : a pytorch module to attach the hook to
                input - list <Tensor> : a list of pytorch tensors, contains the input information to
                            the forward pass
                output - list <Tensor> : a list of pytorch tensors, contains the output information from
                            the forward pass
        """
        try:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        except Exception:
            pass
        mname = module._order
        if self.save:
            if type(output) is not tuple:
                output = [output]
            else:
                output = list(output)
            self.forward_stats[mname] = dict(input=list(input), output=output)

    def backward_hook_fn(self, module, input, output):
        """This is a generic backward hook function which saves all statistics from the backward pass
                during training. This function can be overwritten by a subclass, and the subclass' function
                should be automatically registered in place of this one. 

            This function follows the structure of backward hooks as explained in the pytorch
                docs: https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html

            Args:
                module - torch.nn.Module : a pytorch module to attach the hook to
                input - list <Tensor> : a list of pytorch tensors, contains the input information to
                            the backward pass
                output - list <Tensor> : a list of pytorch tensors, contains the output information from
                            the backward pass
        """
        try:
            pydevd.settrace(suspend=False, trace_only_current_thread=True)
        except Exception:
            pass
        mname = module._order
        if self.save:
            self.backward_stats[mname] = dict(input=list(input), output=list(output))
        if self.backward_return is not None:
            delta = self.backward_return[self.batch_indices, :]
            self.backward_return = None
            if self.save:
                self.backward_stats[mname]["delta"] = delta
            act_input = self.forward_stats[mname]["input"][0]
            grad_weight = act_input.t().mm(delta)
            new_delta = (delta * self.forward_stats[mname]["output"][0]) @ module.weight
            return (new_delta, grad_weight.t())

    def get_hook_fun(self, module):
        """An initial generic hook function for saving gradients.
            Args: 
                module - torch.nn.Module: a pytorch module
        """
        mname = module._order

        def hook_fun(grad):
            if self.save:
                self.grads[mname] = grad

        return hook_fun

    def register_hooks(self, model, layer_names):
        """This function uses the pytorch module's module list
            to grab all layers as specified in a list of layer types.

            Args:
                module - torch.nn.Module: a pytorch module
                layer_names - list<str>: a list of strings of class names for layers to register hooks on
                    e.g. ["Linear"] will register on all linear layers
        """
        modules = list(
            [
                module
                for i, module in enumerate(model.modules())
                if module.__class__.__name__ in layer_names
            ]
        )
        rev_mod = modules[::-1]
        self.orders = dict()
        model.hook = self
        # Iterate through the modules in reverse-order
        for m_i, module in enumerate(rev_mod):
            self.keys.append(str(module) + str(m_i))            
            self.parameters.append(nn.ParameterDict(module.named_parameters()))
            module._order = str(module) + str(m_i)
            module.register_forward_hook(self.forward_hook_fn)
            module.register_backward_hook(self.backward_hook_fn)
            for parameter in module.parameters():
                parameter.register_hook(self.get_hook_fun(module))
            if self.verbose:
                print("Registered hook on module %s" % i)
