import torch
from torch import nn
from types import MethodType
from ._model_fns import train_fn, validation_fn

class GENUINE_E(nn.Module):
    
    def __init__(self, type="resnet34"):
        super().__init__()
        
        self.model = torch.hub.load('pytorch/vision:v0.10.0', type, pretrained=True)
        fc_connections = list(self.model.children())[-1].in_features
        
        self.model = torch.nn.ModuleList([x for x in list(self.model.children())[:-1]])
        
        self.model.append(nn.Linear(fc_connections, 1))
        self.model = self.model.float()
        
        self.train_fn = MethodType(train_fn, self)
        self.validation_fn = MethodType(validation_fn, self)

    #ignore this, only needed to load pickled model
    def train_fn():
        pass
    
    def validation_fn():
        pass   
        
        
    def forward(self, X, ret_fs=False):
        
        for module in self.model[:-1]:
            X = module(X)
            
        if ret_fs:
            return torch.flatten(X, 1)
            
        X = torch.flatten(X, 1)  
              
        return self.model[-1](X)
    
    