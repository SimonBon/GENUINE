import torch
from torch import nn
import numpy as np
from types import MethodType
from ._model_fns import train_fn, validation_fn

EPS = 10e-5

class Baseline(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.thresh = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))
        
        self.train_fn = MethodType(train_fn, self)
        self.validation_fn = MethodType(validation_fn, self)
        
    def train_fn():
        pass
    
    def validation_fn():
        pass    
        
    def forward(self, X):
        
        mycn = X[:, 1]
        nmi = X[:, 0]
        
        mycn_sum = mycn.flatten(1).sum(axis=1)
        nmi_sum = nmi.flatten(1).sum(axis=1) + EPS
        
        ret = mycn_sum / nmi_sum
    
        return ret - self.thresh
    
    
            