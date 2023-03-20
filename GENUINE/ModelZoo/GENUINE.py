from torch import nn
from ..utils.evaluation import get_top_model
import torch
from types import MethodType
from ._model_fns import train_fn, validation_fn
from .GENUINE_E import GENUINE_E
import os

class GENUINE(nn.Module):
    
    def __init__(self, encoder=None, bbox=None, sz=100):
        super().__init__()

        self.sz = sz

        if encoder:
            self.encoder_path = encoder
            self.encoder = torch.load(self.encoder_path, map_location='cpu').float()
        else:
            self.encoder = GENUINE_E().cpu().float()
    
        assert not isinstance(bbox, type(None)), "Please provide a path to a pretrained bounding box model"
        self.bbox_path = bbox
        print(f"Loaded: {self.bbox_path}")
        self.bbox = torch.load(self.bbox_path, map_location='cpu').float()
        
        for param in self.bbox.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(18*self.sz + 512, 100),
            nn.ReLU(),
            nn.Linear(100, 1)).float()
    
        self.train_fn = MethodType(train_fn, self)
        self.validation_fn = MethodType(validation_fn, self)
        
        self.train()
    
    def forward(self, X, ret_fs=False):
        
        encoder_fs = self.encoder(X, ret_fs=True)
        
        box_fs = self._box_return2tensor(self.bbox([x for x in X]), sz=self.sz).detach()

        combined_fs = torch.cat((encoder_fs.flatten(1), box_fs.flatten(1)), 1)
        
        if ret_fs:
            return self.fc(combined_fs), encoder_fs, box_fs
        
        return self.fc(combined_fs)
        
        
    def to(self, d):
        
        self.encoder.to(d)
        self.bbox.to(d)
        self.fc.to(d)
        
        
    def eval(self):
        
        self.encoder.eval()
        self.bbox.eval()
        self.fc.eval()
        
        
    def train(self):
        
        self.encoder.train()
        self.bbox.eval()
        self.fc.train()
    
    
    #ignore this, only needed to load pickled model
    def train_fn():
        pass
    
    def validation_fn():
        pass   
    
    
    def _box_return2tensor(self, ret, sz=100):
        
        return_list = []
        for s in ret:
            
            tensor = torch.cat((s["boxes"], s["labels"].unsqueeze(-1), s["scores"].unsqueeze(-1)), dim=1)
            tensor = tensor[tensor[..., 5] > 0.1]
            
            ret_vec = []
            for i in range(1, 4):
                
                tmp = tensor[tensor[..., 4] == i][:sz]
                tmp = torch.cat((tmp, torch.zeros(sz-tmp.shape[0], 6).to(tensor.device)), dim=0)
                ret_vec.append(tmp)
                
            ret_vec = torch.vstack(ret_vec)
            
            return_list.append(ret_vec)
            
        return torch.stack(return_list)