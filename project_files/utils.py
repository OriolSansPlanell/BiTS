# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:15:04 2024

@author: Dr Oriol Sans Planell
"""

import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        return torch.device("cpu")

def to_device(tensor, device):
    return tensor.to(device)