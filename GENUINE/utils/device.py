import nvidia_smi
import torch

def best_gpu(verbose=False):
    
    if torch.cuda.is_available():
    
        nvidia_smi.nvmlInit()

        max_free = 0
        for i in range(torch.cuda.device_count()):

            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)

            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            
            if verbose:
                print(f"{round(info.free/(10**9), 2)} GB free on cuda:{i}")
            
            if info.free > max_free:
                idx = i
                max_free = info.free
                
        return torch.device(idx)  
            
    else:
        return torch.device("cpu")
            
    