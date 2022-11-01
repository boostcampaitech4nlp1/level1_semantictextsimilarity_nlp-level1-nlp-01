# for utilities
import numpy as np
import random
import torch
import os
import torch.nn.functional as F

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

