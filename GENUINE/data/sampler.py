from torch.utils.data import Sampler
import numpy as np

class custom_sampler(Sampler):
    
    def __init__(self, data_source):
        super().__init__(data_source=data_source)
        self.data_source = data_source
        self.mix = self.data_source.label_mix
        self.keys = self.data_source.keys
        
        self.pos = [n for n, (k, v) in enumerate(self.mix.items()) if v!=0]
        self.neg = [n for n, (k, v) in enumerate(self.mix.items()) if v==0]
        
                
    def __iter__(self):
        
        idxs = self.neg.copy()

        added = self.label_informed_sampler(self.mix, len(self.neg))
                      
        diff = len(added) - len(idxs) 
        
        idxs.extend(list(np.random.choice(self.neg, diff)))
        
        idxs.extend(added)
        
        np.random.shuffle(idxs)
        
        return iter([int(i) for i in idxs])
        
        
    def __len__(self):
        
        return 2*len(self.label_informed_sampler(self.mix, len(self.neg)))
    
    def label_informed_sampler(self, mix, out_sz):
        
        match = {n: v for n, (k, v) in enumerate(mix.items()) if v!=0}
        
        occ = sum([v for k, v in match.items()])
        
        match = {k: np.ceil(v*out_sz/occ).astype(int) for k, v in match.items()}
            
        ret = []
        for k,v in match.items():
            ret.extend([k]*v)
        
        return ret