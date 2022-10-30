# for utilities
import numpy as np
import random
import torch
import os

def optimizer_selector(name, params, lr):
    
    if name == "Adam":
        optimizer = torch.optim.Adam(params=params, lr=lr)
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(params=params,lr=lr)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(params=params,lr=lr)
    elif name == 'RAdam':
        optimizer = torch.optim.RAdam(params=params,lr=lr)

    return optimizer
