import math
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer
import sys


class Adam(Optimizer):
    def __init__(self, params, svd=False, thres=1.001, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, thres=thres)
        super(Adam, self).__init__(params, defaults)

        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)
        self.sparsed_channel_indices = defaultdict(dict)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('svd', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            svd = group['svd']
            for name, p in zip(group['name'], group['params']):
                if p.grad is None:
                    continue
                if 'adapter' in name:
                    name = name.replace('adapter', 'conv')
                name = f"backbone.{name}"
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                update = self.get_update(group, grad, p)
                if svd and len(self.transforms) > 0:
                    if len(update.shape) == 4:
                        squeezed_update = update.squeeze()  
                        indice_sparsed_channel = self.sparsed_channel_indices[name] 
                        update_mask = torch.zeros(squeezed_update.size(1), dtype=bool) 
                        update_mask[indice_sparsed_channel] = 1 
                        masked_update = squeezed_update[:, update_mask]
                        update_ = torch.mm(masked_update.view(masked_update.size(0), -1), self.transforms[name]).view_as(masked_update)

                        transformed_update = torch.zeros_like(squeezed_update) 
                        transformed_update[:, update_mask] = update_ 
                        transformed_update = transformed_update.unsqueeze(2).unsqueeze(3)
                        p.data.add_(transformed_update)       
                    else:
                        update_ = torch.mm(update, self.transforms[name])
                        p.data.add_(update_)            
                else:
                    update_ = update
                    p.data.add_(update_)
        return loss

    def get_sparsed_channel_indices(self, sparsed_channel_indices):
        self.sparsed_channel_indices = sparsed_channel_indices


    def get_eigens(self, masked_covar_matrix):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for name, p in zip(group['name'], group['params']):
                if 'adapter' in name:
                    name = name.replace('adapter', 'conv')
                name = f"backbone.{name}"
                eigen = self.eigens[name]
                _, eigen_value, eigen_vector = torch.svd(masked_covar_matrix[name], some=False)
                eigen['eigen_value'] = eigen_value
                eigen['eigen_vector'] = eigen_vector
    

    def get_transforms(self):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for name, p in zip(group['name'], group['params']):
                if 'adapter' in name:
                    name = name.replace('adapter', 'conv')
                name = f"backbone.{name}"
                ind = self.eigens[name]['eigen_value'] <= self.eigens[name]['eigen_value'][-1] * group['thres']
                basis = self.eigens[name]['eigen_vector'][:, ind]
                transform = torch.mm(basis, basis.transpose(1, 0))
                self.transforms[name] = transform / torch.norm(transform)
                self.transforms[name].detach_()

    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]

        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)

        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * \
            math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom
        return update
