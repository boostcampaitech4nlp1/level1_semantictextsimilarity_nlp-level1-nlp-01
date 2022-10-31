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
        optimizer = torch.optim.AdamW(params=params,lr=lr)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(params=params,lr=lr)
    elif name == 'RAdam':
        optimizer = torch.optim.RAdam(params=params,lr=lr)

    return optimizer

def loss_fct_selector(name):

    loss_func = torch.nn.L1Loss()

    if name == 'L1Loss':
        loss_func = torch.nn.L1Loss()

    elif name == 'HuberLoss':
        loss_func = torch.nn.HuberLoss()

    elif name == 'SmoothL1Loss':
        loss_fun = torch.nn.SmoothL1Loss()
    return loss_func

if __name__ == '__main__':

    print("This is utils.py")
