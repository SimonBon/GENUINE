import os
import h5py
import torch
import pandas as pd
from torch.utils.data import DataLoader
from ..data.datasets import PatchDataset
from ..data.custom_transforms import ToTensor, ManualIntensity, ManualBlur, ManualNoise
from .device import best_gpu
from tqdm import tqdm
import numpy as np

def get_top_model(base):
    
    state_dict_paths = [os.path.join(base, x) for x in os.listdir(base) if ".pt" in x.lower()]
    state_dict_dict = []
    for state_dict_path in state_dict_paths:
        
        state_dict = torch.load(state_dict_path, map_location="cpu")
        state_dict_dict.append({"validation_loss": state_dict["validation_loss"],
                                "path": state_dict_path})
        
    state_dict_df = pd.DataFrame(state_dict_dict)
    state_dict_df = state_dict_df[state_dict_df['validation_loss'].notna()]
    state_dict_df = state_dict_df.sort_values(by="validation_loss", ascending=False)

    best_model = state_dict_df.iloc[-1]
    return best_model["path"]
    
    
def evaluate_model_dilutions(model_dir, dataset_path="/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/SAMPLE_TIFS/H5_EVAL/DILUTIONS.h5", n=None, samples=None):

    if isinstance(samples, type(None)):

        with h5py.File(dataset_path, 'r') as f:
            
            samples = list(f.keys())
        
    model = torch.load(get_top_model(model_dir))["model"].cpu()
    model.eval()
    device = best_gpu()
    model.to(device)
        
    transform = [ToTensor()]
       
    results_dict = {} 
    for sample in samples:
        
        ds = PatchDataset(dataset_path, dataset=sample, transform=transform, n=n)
        dl = DataLoader(ds, batch_size=32, num_workers=4)
        
        preds = []
        for X, _ in tqdm(dl):
            
            ret = model(X.to(device)).detach().cpu()
            preds.extend(list(ret > 0))
            
        #print(f"{sample}: {torch.sum(torch.tensor(preds))/len(preds):%}")
        results_dict[sample] = (torch.sum(torch.tensor(preds))/len(preds)).item()
        
    return results_dict


def evaluate_model_testset(model_dir, dataset_path="/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/SAMPLE_TIFS/H5/DATASET.h5", n=None, sample="test"):
    
    model = torch.load(get_top_model(model_dir))["model"].cpu()
    model.eval()
    device = best_gpu()
    model.to(device)
        
    transform = [ToTensor()]
       
    results_dict = {} 

    ds = PatchDataset(dataset_path, dataset=sample, transform=transform, n=n)
    dl = DataLoader(ds, batch_size=32, num_workers=4)
    
    preds, targets = [], []
    for X, y in tqdm(dl):
        
        ret = model(X.to(device)).detach().cpu()
        preds.extend(list(ret > 0))
        targets.extend(list(y))
      
    targets = np.array(targets)
    preds = np.array(preds)  
    
    TP = len(np.where(np.logical_and(targets == 1, preds == 1))[0])
    TN = len(np.where(np.logical_and(targets == 0, preds == 0))[0])
    FP = len(np.where(np.logical_and(targets == 0, preds == 1))[0])
    FN = len(np.where(np.logical_and(targets == 1, preds == 0))[0])
    SPEC = np.round(TN / (TN + FP)*100,2)
        
    precision = np.round(TP/(TP+FP)*100,2)
    recall = np.round(TP/(TP+FN)*100,2)
    F1 = np.round(2*precision*recall/(precision+recall),2)
    ACC = ((TP + TN) / (TP+TN+FN+FP))*100
            
    results_dict = {
        "TPR":  TP/ (TP + FN),
        "TNR": TP/ (TP + FN),
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision":precision,
        "recall": recall,
        "F1": F1,
        "specificity": SPEC,
        "sensitivity": recall,
        "accuracy": ACC
    }
        
    return results_dict


def evaluate_model_quality(model_dir, dataset_path="/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/SAMPLE_TIFS/H5_EVAL/DILUTIONS.h5", n=None, samples={"S4": 1, "S12": 0}):
    
    model = torch.load(get_top_model(model_dir))["model"].cpu()
    model.eval()
    device = best_gpu()
    model.to(device)
        
    brightness_conditions = [[0.5, 0.5, 0.5], [1, 1, 1], [2, 2, 2], [4, 4, 4]]
    blurr_conditions = [[0,0,0], [1, 1, 1]]
    
    ret_dict = {}
   
    for r0,g0,b0 in brightness_conditions:
    
        for r1, g1, b1 in blurr_conditions:
            
            preds, targets = [], []
            for sample, y in samples.items():

                transform = [ToTensor(), ManualIntensity((r0,g0,b0)), ManualBlur((r1, g1, b1))]
            
                ds = PatchDataset(dataset_path, dataset=sample, transform=transform, n=n)
                dl = DataLoader(ds, batch_size=32, num_workers=4)
                
                for X, _ in tqdm(dl):
                    
                    ret = model(X.to(device)).detach().cpu()
                    preds.extend(list((ret > 0).numpy().squeeze()))
                    targets.extend([y]*32)
                
            targets = np.array(targets)
            preds = np.array(preds)  
            
            TP = len(np.where(np.logical_and(targets == 1, preds == 1))[0])
            TN = len(np.where(np.logical_and(targets == 0, preds == 0))[0])
            FP = len(np.where(np.logical_and(targets == 0, preds == 1))[0])
            FN = len(np.where(np.logical_and(targets == 1, preds == 0))[0])
            SPEC = np.round(TN / (TN + FP)*100,2)
        
            try:  
                precision = np.round(TP/(TP+FP)*100,2)
            except:
                precision = 0
                
            recall = np.round(TP/(TP+FN)*100,2)
            F1 = np.round(2*precision*recall/(precision+recall),2)
            ACC = ((TP + TN) / (TP+TN+FN+FP))*100
                    
            results_dict = {
                "TPR":  100 *TP/ (TP + FN) ,
                "TNR": 100 * TN/ (TN + FP),
                "TP": TP,
                "TN": TN,
                "FP": FP,
                "FN": FN,
                "precision":precision,
                "recall": recall,
                "F1": F1,
                "specificity": SPEC,
                "sensitivity": recall,
                "accuracy": ACC
            }
            
            ret_dict[f"BR{r0}{g0}{b0}_BL{r1}"] = results_dict
        
    return ret_dict
