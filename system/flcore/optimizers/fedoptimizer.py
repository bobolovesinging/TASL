import torch
from torch.optim.optimizer import Optimizer

class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_cs, client_cs):
        for group in self.param_groups:
            for p, sc, cc in zip(group['params'], server_cs, client_cs):
                p.data.add_(other=(p.grad.data + sc - cc), alpha=-group['lr'])


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)

# class PerturbedGradientDescent(Optimizer):
#     def __init__(self, params, lr=0.01, mu=0.0):
#         default = dict(lr=lr, mu=mu)
#         super().__init__(params, default)

#     @torch.no_grad()
#     def step(self, global_params, device):
#         for group in self.param_groups:
#             for p, g in zip(group['params'], global_params):
#                 g = g.to(device)
#                 d_p = p.grad.data + group['mu'] * (p.data - g.data)
#                 p.data.add_(d_p, alpha=-group['lr'])

# class PerturbedGradientDescent(Optimizer):
#     def __init__(self, params, lr=0.01, mu=0.0):
#         default = dict(lr=lr, mu=mu)
#         super().__init__(params, default)

#     @torch.no_grad()
#     def step(self, global_params, old_params, device):
#         for group in self.param_groups:
#             for p, g, o in zip(group['params'], global_params, old_params):
#                 g = g.to(device)
#                 o = o.to(device)
#                 d_p = p.grad.data + 0.0001 * (p.data - g.data)**2 - 0.0001 * (p.data - o.data)**2
#                 p.data.add_(d_p, alpha=-group['lr'])

class PerturbedGradientDescenth(Optimizer):
    def __init__(self, params, lr=0.01, nu=0.0):
        default = dict(lr=lr, nu=nu)
        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, old_params, device):
        for group in self.param_groups:
            for p, g, o in zip(group['params'], global_params, old_params):
                g = g.to(device)
                o = o.to(device)
                d_p = p.grad.data + 0.0001 * (p.data - g.data)**2 - 0.0001 * (p.data - o.data)**2
                p.data.add_(d_p, alpha=-group['lr'])




class PerturbedGradientDescent(Optimizer):
    """Implementation of Perturbed Gradient Descent, i.e., FedProx optimizer in PyTorch."""
    
    def __init__(self, params, learning_rate=0.001, mu=0.01):
        if learning_rate <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if mu < 0.0:
            raise ValueError("Invalid proximal term coefficient (mu): {}".format(mu))

        defaults = {'lr': learning_rate, 'mu': mu}
        super(PerturbedGradientDescent, self).__init__(params, defaults)

    def step(self, global_params):
        """
        Perform a single optimization step.

        Args:
            global_params (list): List of global parameters (vstar in TensorFlow) to use for the proximal term.
        """
        loss = None

        # Ensure that the length of model params and global_params are the same
        for group, global_param in zip(self.param_groups, global_params):
            mu = group['mu']
            lr = group['lr']

            # Loop through each parameter and apply the update
            for p, g in zip(group['params'], global_param):
                if p.grad is None:
                    continue
                
                # Proximal gradient update: p = p - lr * (grad + mu * (p - g))
                d_p = p.grad.data + mu * (p.data - g.data)
                p.data = p.data - lr * d_p

        return loss
    
    def set_params(self, global_params, model):
        """
        Set the global parameters (vstar) for the model.

        Args:
            global_params (list): List of global parameters to set.
            model (torch.nn.Module): The PyTorch model on which to set the parameters.
        """
        for param, global_param in zip(model.parameters(), global_params):
            param.data.copy_(global_param.data)
