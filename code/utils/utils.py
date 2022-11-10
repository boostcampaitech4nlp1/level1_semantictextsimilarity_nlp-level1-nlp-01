import os
import random

import importlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from typing import Union, Callable
from itertools import count 

# Loss function selector
def load_obj(obj_path: str, default_obj_path: str=''):
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)

# Optimizer selector
def optimizer_selector(name, params, lr):
    if name == "Adam":
        optimizer = torch.optim.Adam(params=params, lr=lr)
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(params=params, lr=lr)
    elif name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=params, lr=lr)
    elif name == 'NAdam':
        optimizer = torch.optim.NAdam(params=params, lr=lr)
    elif name == 'RAdam':
        optimizer = torch.optim.RAdam(params=params, lr=lr)

    return optimizer

# For smart Loss
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d

def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)

class SMARTLoss(nn.Module):
    
    def __init__(
        self,
        eval_fn: Callable,
        loss_fn: Callable,
        loss_last_fn: Callable = None, 
        norm_fn: Callable = inf_norm, 
        num_steps: int = 1,
        step_size: float = 1e-3, 
        epsilon: float = 1e-6,
        noise_var: float = 1e-5
    ) -> None:
        super().__init__()
        self.eval_fn = eval_fn 
        self.loss_fn = loss_fn
        self.loss_last_fn = default(loss_last_fn, loss_fn)
        self.norm_fn = norm_fn
        self.num_steps = num_steps 
        self.step_size = step_size
        self.epsilon = epsilon 
        self.noise_var = noise_var
        
    def forward(self, embed: Tensor, state: Tensor) -> Tensor:
        noise = torch.randn_like(embed, requires_grad=True) * self.noise_var

        # Indefinite loop with counter 
        for i in count():
            # Compute perturbed embed and states 
            embed_perturbed = embed + noise 
            state_perturbed = self.eval_fn(embed_perturbed)
            # Return final loss if last step (undetached state)
            if i == self.num_steps: 
                return self.loss_last_fn(state_perturbed, state) 
            # Compute perturbation loss (detached state)
            loss = self.loss_fn(state_perturbed, state.detach())
            # Compute noise gradient ∂loss/∂noise
            noise_gradient, = torch.autograd.grad(loss, noise)
            # Move noise towards gradient to change state as much as possible 
            step = noise + self.step_size * noise_gradient 
            # Normalize new noise step into norm induced ball 
            step_norm = self.norm_fn(step)
            noise = step / (step_norm + self.epsilon)
            # Reset noise gradients for next step
            noise = noise.detach().requires_grad_()

def soft_voting(model, model_names, cfg):
    models = torch.nn.ModuleDict()
    roberta_dataloader, roberta_trainer = get_trainer_dataloader('klue/roberta-large', cfg)
    tunib_dataloader, tunib_trainer = get_trainer_dataloader('tunib/electra-ko-en-base', cfg)
    
    for i,name in enumerate(model_names):
        if '.ckpt' in name:
            models[get_name(name,i)] = model
        elif '.pt' in name:
            models[get_name(name,i)] = torch.load(f'./models/{name}')

    predictions = []
    for model_name in models:
        model = models[model_name]
        if 'tunib' in name:
            predict = tunib_trainer.predict(model=model, datamodule=tunib_dataloader)
        elif 'roberta' in name:
            predict = roberta_trainer.predict(model=model, datamodule=roberta_dataloader)
        predict = list(float(i) for i in torch.cat(predict))
        predictions.append(predict)

    vote_predictions = np.sum(np.array(predictions), axis=0)/len(predictions)
    vote_predictions = torch.from_numpy(vote_predictions)
    vote_predictions = list(round(float(i), 1) for i in vote_predictions)
    
    return vote_predictions

def weighted_voting(model, model_names, weights, cfg):
    models = torch.nn.ModuleDict()
    roberta_dataloader, roberta_trainer = get_trainer_dataloader('klue/roberta-large', cfg)
    tunib_dataloader, tunib_trainer = get_trainer_dataloader('tunib/electra-ko-en-base', cfg)
    
    for i,name in enumerate(model_names):
        if '.ckpt' in name:
            models[get_name(name,i)] = model
        elif '.pt' in name:
            models[get_name(name,i)] = torch.load(f'./models/{name}')

    predictions = []
    for idx, model_name in enumerate(models):
        model = models[model_name]
        if 'tunib' in name:
            predict = tunib_trainer.predict(model=model, datamodule=tunib_dataloader)
        elif 'roberta' in name:
            predict = roberta_trainer.predict(model=model, datamodule=roberta_dataloader)
        predict = list(float(i)*weights[idx] for i in torch.cat(predict))
        predictions.append(predict)

    vote_predictions = np.sum(np.array(predictions), axis=0)/sum(weights)
    vote_predictions = torch.from_numpy(vote_predictions)
    vote_predictions = list(round(float(i), 1) for i in vote_predictions)
    
    return vote_predictions
