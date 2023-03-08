from torch.utils.data import Dataset
from ..utils.data import Range
import os
import torch
import h5py
import torch
import numpy as np
from .custom_transforms import *
from torchvision.transforms import ToTensor, transforms
from typing import Union


DEFAULT_TRANSFORMS = [ToTensor(), RandomAffine(scale=[0.3, 1.2]), RandomFlip(), RandomNoise(), RandomIntensity(0.8, 2.5), RandomChannelSkip([0.8, 0, 0.8])]
DEFAULT_FRCNN_TRANSFORMS = [ToTensorBoxes(), RandomBoxRotation(), RandomBoxFlip(), RandomBoxNoise(), RandomBoxIntensity(0.8, 2.5), RandomBoxChannelSkip([0.5, 0.5, 0.5])]


COLOR_IDX = {"red": 0, "green": 1, "blue": 2}
DEFAULT_CHANNELS = ["red", "green", "blue"]


class PatchDataset(Dataset):
    
    def __init__(self, dataset_path: Union[os.PathLike, str], dataset: str, channels: list=DEFAULT_CHANNELS, transform: list=DEFAULT_TRANSFORMS, n: int=None, ret_id=False):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.transforms = transforms.Compose(transform)
        self.h5py_file = h5py.File( self.dataset_path, 'r')
        # self._channels = self._check_channels(channels)
        self.n = n
        self.ret_id = ret_id
        
        self._r = r = Range(len(self))
        
    def __getitem__(self, subscript):
        
        if isinstance(subscript, slice):
            return [self._return_idxed(i) for i in self._r[subscript]]
        
        elif isinstance(subscript, int):
            return self._return_idxed(subscript)
        
        elif isinstance(subscript, (list, np.ndarray)):
            
            if isinstance(subscript[0], (np.bool_, bool)):
                idxs = np.where(subscript)[0]
            else:
                idxs = subscript
            return [self._return_idxed(i) for i in idxs]
        else:
            raise Exception(f"{type(subscript)} is not a valid index")
    
    def __len__(self):
        if self.n:
            return self.n
        return self.h5py_file.get(f"{self.dataset}/X").shape[0]

    def _return_idxed(self, idx):
        
        X = self.h5py_file.get(f"{self.dataset}/X")[idx]
        y = torch.tensor(self.h5py_file.get(f"{self.dataset}/y")[idx])
    
        #self._adapt_channels(X)
            
        if self.ret_id:
            
            return self.transforms(X).float(), y, torch.tensor(self.h5py_file.get(f"{self.dataset}/id")[idx])
        
        else:
            
            return  self.transforms(X).float(), y
    
    # def _get_channel_size(self):
    #     return self.h5py_file.get(f"{self.dataset}/X").shape[-1]
    
    
    # def _check_channels(self, channels):
        
    #     idxs = []
    #     for channel in channels:
            
    #         if isinstance(channel, str):
    #             if channel in COLOR_IDX:
                    
    #                 idxs.append(COLOR_IDX[channel])
                    
    #             else:
                    
    #                 raise ValueError
                
    #         elif isinstance(channel, int):
                
    #             idxs.append(channel)
                
    #     return list(set(range(self._get_channel_size())) - set(idxs))
        
    
    # def _adapt_channels(self, X):
        
    #     for c in self._channels:
            
    #         X[..., c] = np.zeros_like(X[..., c])

    #     return 
    
    
# class BoxesDataset(Dataset):
    
#     def __init__(self, dataset_path: os.PathLike, dataset: str, n:int=None):
#         super().__init__()
#         self.dataset_path = dataset_path
#         self.dataset = dataset
#         self.h5py_file = h5py.File( self.dataset_path, 'r')
#         self.tt = ToTensor()
#         self.n = n
        
#         self.data = self.get_data()
        
#     def __len__(self):
            
#         if self.n:
#             return self.n
#         else:
#             return self.data["X"].shape[0]
        
#     def get_data(self):
        
#         with  h5py.File( self.dataset_path, 'r') as fin:
            
#             X = np.array(fin[self.dataset]["X"]).astype(np.float32)
#             y = np.array(fin[self.dataset]["y"]).astype(np.float32)
            
#         return {"X": torch.tensor(X), "y": torch.tensor(y)}
        
#     def __getitem__(self, idx):
            
#         return self.data["X"][idx], self.data["y"][idx]
    
    

class FRCNN_MYCN(Dataset):
    
    def __init__(self, dataset_path: os.PathLike, dataset: str, transform:list = DEFAULT_FRCNN_TRANSFORMS, n=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = dataset
        self.transform = transform
        self.h5py_file = h5py.File( self.dataset_path, 'r')
        self.keys = self._keys()
        self.n = n
        self.label_mix = self.get_label_mix()
        
    def __getitem__(self, idx):
         
        image = self.h5py_file[self.dataset][self.keys[idx]]["image"][()]
        boxes = self.h5py_file[self.dataset][self.keys[idx]]["boxes"][()]
        labels = self.h5py_file[self.dataset][self.keys[idx]]["labels"][()]
        
        package = (image, boxes, labels)
        for trans in self.transform:
            package = trans(*package)
            
        image, boxes, labels = package  
        
        assert boxes.shape[0] == labels.shape[0], "labels and boxes dont match"
        
        return {"image": image.float(), "boxes": boxes.long(), "labels": labels.long()}
    
    def __len__(self):
        
        return len(self.keys)
    
    def _keys(self):
        
        return list(self.h5py_file[self.dataset].keys())
    
    def get_label_mix(self):
        
        mix = {}
        for k, v in self.h5py_file[self.dataset].items():
            
            unique, counts = np.unique(v["labels"], return_counts=True)
            d = dict(zip(unique, counts))
            
            try:
                mix[k] = d[3]
                
            except:
                mix[k] = 0 
                
        return mix
    
        # y = torch.tensor(self.h5py_file.get(f"{self.dataset}/y")[idx])
    
        # self._adapt_channels(X)
            
        # if self.ret_id:
            
        #     return self.transforms(X).double(), y, torch.tensor(self.h5py_file.get(f"{self.dataset}/id")[idx])
        
        # else:
            
        #     return  self.transforms(X).double(), y
            
        #     if len(boxes) != 0:
        #         if self.use_transform:
                
        #             for transform in self.transforms:
        #                 image, boxes, labels = transform(image, boxes, labels)                
                        
        #         return  {"image": image, "boxes": boxes, "labels": labels}
        
        #     else:
                
        #         #benötigt wenn durch die wahl der rgb layer keine boxen mehr vorhanden sind
        #         return  {"image": image, "boxes": torch.tensor([[0, 0, image.shape[-1], image.shape[-2]]]), "labels": torch.tensor([0])}
    