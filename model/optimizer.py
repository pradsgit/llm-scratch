import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import math

# should subclass torch.optim.Optimizer
class SGD(torch.optim.Optimizer): 
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f'invalid learning rate: {lr}')
        defaults = {'lr': lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                # update p.data
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data # gradient of loss wrt to p
                p.data -= lr / math.sqrt(t+1) * grad
                state['t'] = t + 1

        return loss


# implement AdamW
class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8, 
        weight_decay=None,
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)
        

    def step(self, closure: Optional[Callable]=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # get that group's lr and other hyperparams
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    # do nothing if there is no gradient value
                    continue

                # update p.data with Adam
                state = self.state[p]
                grad = p.grad.data
                t = state.get('t', 1)
                
                # calculate biased first momentum

                if 'first_moment' not in state:
                    state['first_moment'] = torch.zeros_like(p.data)
                    state['second_moment'] = torch.zeros_like(p.data)
                    state['t'] = 1

                first_moment = state['first_moment']
                second_moment = state['second_moment']

                # upadte first moment estimate
                state['first_moment'] = betas[0] * first_moment + (1.0 - betas[0]) * grad
                state['second_moment'] = betas[1] * second_moment + (1.0 - betas[1]) * grad * grad

                # adjusted learning for timestep t
                updated_lr = lr * (math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t))

                p.data -= updated_lr * (state['first_moment'] / (torch.sqrt(state['second_moment']) + eps))
                if weight_decay is not None:
                    p.data -= lr * weight_decay * p.data
                state['t'] = t + 1

        return loss



