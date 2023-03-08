import torch
from torch import nn
from ..utils.evaluation import get_top_model
from ._model_fns import train_fn, validation_fn
from types import MethodType
import os

class GENUINE_B(nn.Module):
    
    def __init__(self, bbox=None, input_size=6, hidden_size=128, output_size=1):
    
        super().__init__()
        
        assert not isinstance(bbox, type(None)), "Please provide a path to a pretrained bounding box model"
        self.bbox_path = bbox
        self.bbox = torch.load(self.bbox_path, map_location="cpu").float()
        
        for param in self.bbox.parameters():
            param.requires_grad = False
        
        self.bbox.eval()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True).float()
        self.fc = nn.Linear(hidden_size, output_size).float()
        
        self.train_fn = MethodType(train_fn, self)
        self.validation_fn = MethodType(validation_fn, self)
        
        
    def forward(self, X, ret_fs=False):
                    
        X = self.bbox(X)
    
        X = self._box_return2tensor(X)

        if ret_fs:
            fs = X.clone().cpu().detach()
            
        X, _ = self.lstm(X)

        X = self.fc(X[:, -1])
        
        if ret_fs:
            return X.squeeze(), fs
        
        return X.squeeze()
    
    
    def train(self):
            
        self.bbox.eval()
        self.lstm.train()
        self.fc.train()
    
    
    def eval(self):
        
        self.bbox.eval()
        self.lstm.eval()
        self.fc.eval()
    
    
    def to(self, device):
        
        self.bbox.to(device)
        self.lstm.to(device)
        self.fc.to(device)
        
    
    def _box_return2tensor(self, X):
    
        return_list = []
        for s in X:
            
            tensor = torch.cat((s["boxes"], s["labels"].unsqueeze(-1), s["scores"].unsqueeze(-1)), dim=1)[:60]
            
            if tensor.shape[0] != 60:
            
                tensor = torch.cat((tensor, torch.zeros(60-tensor.shape[0], 6).to(tensor.device)), dim=0)
            
            return_list.append(tensor.T)
            
        return torch.stack(return_list).permute(0, 2, 1)
            
    #ignore this, only needed to load pickled model
    def train_fn():
        pass
    
    def validation_fn():
        pass  
    
    