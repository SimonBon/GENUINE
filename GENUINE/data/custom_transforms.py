from torch import nn
import numpy as np
from torchvision import transforms
import torch
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, GaussianBlur


CHANNEL_LABEL_CORRESPONDANCE = {0: [1], 1: [2, 3]}
  
class RandomAffine(torch.nn.Module):
        
    def __init__(self, degrees=(0,360), translate=(0, 0.1), scale=(0.5, 1.2), fill=0):
        super().__init__()
        self.transformer = transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale, fill=fill)

    def __call__(self, tensor):
        
        return self.transformer(tensor)
  
  
class RandomFlip(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.vert = transforms.RandomVerticalFlip()
        self.hor = transforms.RandomHorizontalFlip()
    
    def __call__(self, tensor):
        
        tensor = self.vert(tensor)
        tensor = self.hor(tensor)
        
        return tensor
    
    
class RandomNoise(torch.nn.Module):
    
    def __init__(self, mean=[0, 0], std=[0, 0.07]):
        
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        
        mean = np.random.uniform(self.mean[0], self.mean[1],1)[0]
        std = np.random.uniform(self.std[0], self.std[1], 1)[0]

        noise = torch.normal(mean, std, img.shape)
        
        noise[img==0] = 0

        if np.logical_or(img[2]==0, img[2]==1).all():
            noise[2] = 0
    
        return torch.clip(img + noise, 0, 1)
    
    
class RandomIntensity(torch.nn.Module):
    
    def __init__(self, lower_scale_limit=1/2, upper_scale_limit=2):
        super().__init__()
        
        self.lower_scale_limit = lower_scale_limit
        self.upper_scale_limit = upper_scale_limit
        
    def __call__(self, tensor: torch.Tensor):
        
        scale_val = np.random.uniform(
            self.lower_scale_limit, 
            self.upper_scale_limit, 
            3
        )
 
        for idx, s_val in enumerate(scale_val):
            
            #dann is es eine maske oder der channel istz komplett schwarz
            if np.logical_or(tensor[idx]==0, tensor[idx]==1).all():
                continue
            
            tensor[idx] = tensor[idx]*(s_val)
            torch.clip(tensor, 0, 1, out=tensor)
                
        return tensor
    

class RandomChannelSkip(torch.nn.Module):

    def __init__(self, ps=[0.5, 0, 0.5]):
        super().__init__()
        
        self.skip_prob = torch.tensor(ps)
        
    def __call__(self, tensor: torch.Tensor):
        
        skip = torch.rand(3) <= self.skip_prob

        tensor[skip] = torch.zeros(tensor.shape[1:], dtype=tensor.dtype)
        
        return tensor
    
    
class RandomBoxFlip(nn.Module):
    
    def __init__(self, ps=(0.5, 0.5)):
        super().__init__()
        self.ps = ps
        self.vert = transforms.RandomVerticalFlip(p=1)
        self.hor = transforms.RandomHorizontalFlip(p=1)
        
    def __call__(self, image, boxes, labels):
        
        boxes = boxes.clone()
        image = image.clone()
        var = np.random.uniform(0,1,2)
        if var[0] <= self.ps[0]:
            image = self.vert(image)
            boxes[:,1] = -boxes[:,1]+image.shape[1]
            boxes[:,3] = -boxes[:,3]+image.shape[1]
            
        if var[1] <= self.ps[1]:
            image = self.hor(image)
            boxes[:,0] = -boxes[:,0]+image.shape[2]
            boxes[:,2] = -boxes[:,2]+image.shape[2]
            
        ret_boxes = []  
        ret_labels = []  
        for box, label in zip(boxes, labels):
            if max(box[0], box[2]) - min(box[0], box[2]) > 0 and max(box[1], box[3]) - min(box[1], box[3]) > 0:
                ret_boxes.append([min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])])
                ret_labels.append(label)

        return image, torch.tensor(ret_boxes), torch.tensor(ret_labels)
    
class RandomBoxRotation(nn.Module):
    
    def __init__(self, degrees=(0,360), translate=(0,0.1), scale=(0.5, 1.2), interpolation=InterpolationMode.NEAREST):
        super().__init__()
        
        self.affine = transforms.RandomAffine(degrees=degrees, translate=translate, scale=scale, interpolation=interpolation)
        
    def __call__(self, image, boxes, labels):
        
        boxes = boxes.clone()
        image = image.clone()
        box_ims = []
        for box in boxes:
            box_im = torch.zeros_like(image)[0]
            box_im[box[1]:box[3], box[0]:box[2]] = 1
            box_ims.append(box_im)
        box_ims = torch.stack(box_ims)
        tmp = torch.cat((image, box_ims))
        image = self.affine(tmp)
        
        boxes = image[3:]
        image = image[:3]
        
        boxes_coords, skip_idxs = self.get_box_coords(boxes)

        ret_boxes = []    
        for box in boxes_coords:
            ret_boxes.append([min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3])])
            
        boxes_coords = torch.tensor(ret_boxes)
        
        keep = list(set(range(len(labels))) - set(skip_idxs))
    
        return image, boxes_coords, labels[keep]
        

    def get_box_coords(self, box_ims):
        
        box_ims.bool()
        boxes = []
        skip_idxs = []
        for n, box in enumerate(box_ims):
            idxs = torch.where(box)
            if len(idxs[0]) == 0:
                skip_idxs.append(n)
                continue
            y = idxs[0].min(), idxs[0].max()
            x = idxs[1].min(), idxs[1].max()
            boxes.append([int(x[0]), int(y[0]), int(x[1]), int(y[1])])
            
        return torch.tensor(boxes), skip_idxs
    
    
class RandomBoxIntensity(torch.nn.Module):
    
    def __init__(self, lower_scale_limit=1/2, upper_scale_limit=2):
        super().__init__()
        
        self.lower_scale_limit = lower_scale_limit
        self.upper_scale_limit = upper_scale_limit
        
    def __call__(self, tensor: torch.Tensor, box: torch.Tensor, labels):
        
        scale_val = np.random.uniform(
            self.lower_scale_limit, 
            self.upper_scale_limit, 
            3
        )
             
        for i in range(3):
            if np.logical_or(tensor[i]==0, tensor[i]==1).all():
                scale_val[i] = 0
                            
        for idx, s_val in enumerate(scale_val):
            
            tensor[idx] = tensor[idx]*(s_val)
            torch.clip(tensor, 0, 1, out=tensor)
                
        return tensor, box, labels


class RandomBoxNoise(torch.nn.Module):
    
    def __init__(self, mean=[0, 0], std=[0, 0.07]):
        
        self.mean = mean
        self.std = std
    
    def __call__(self, img, box, labels):
        
        mean = np.random.uniform(self.mean[0], self.mean[1],1)[0]
        std = np.random.uniform(self.std[0], self.std[1], 1)[0]

        random_weight = 0.1*torch.rand(1)
        noise = random_weight * torch.normal(mean, std, img.shape)

        noise[img==0] = 0

        if np.logical_or(img[2]==0, img[2]==1).all():
            noise[2] = 0

        return torch.clip(img + noise, 0, 1), box, labels


class RandomBoxChannelSkip(torch.nn.Module):
    
    def __init__(self, ps=[0.5, 0, 0.5]):
        super().__init__()
        
        self.skip_prob = torch.tensor(ps)
        
    def __call__(self, tensor: torch.Tensor, boxes: torch.Tensor, labels):
        
        skip = torch.rand(3) <= self.skip_prob

        tensor[skip] = torch.zeros(tensor.shape[1:], dtype=tensor.dtype)
        
        skipped_idx = []
        for i in torch.arange(3)[skip].numpy():
            
            if i in CHANNEL_LABEL_CORRESPONDANCE:
                skipped_idx.extend([n for n,x in enumerate(labels) if x in CHANNEL_LABEL_CORRESPONDANCE[i]])
    
        keep = list(set(range(len(labels))) - set(skipped_idx))

        return tensor, boxes[keep], labels[keep]

class ToTensorBoxes(torch.nn.Module):
    
    def __init__(self):
            super().__init__()
            
            self.tt = ToTensor()
        
    def __call__(self, tensor: torch.Tensor, boxes: torch.Tensor, labels):
    
        return self.tt(tensor), torch.tensor(boxes), torch.tensor(labels)


class ManualIntensity(torch.nn.Module):
    
    def __init__(self, scale_vals=[1,1,1]):
        super().__init__()
        
        self.scale_vals = scale_vals
        
    def __call__(self, tensor: torch.Tensor):
        
        for idx, s_val in enumerate(self.scale_vals):
            
            tensor[idx] = tensor[idx]*(s_val)
            torch.clip(tensor, 0, 1, out=tensor)
                
        return tensor

    
class ManualNoise(torch.nn.Module):
    
    def __init__(self, sigma=0):
        super().__init__()
        
        eps = 10e-5
        self.sigma = sigma + eps
        
    def __call__(self, tensor: torch.Tensor):
        
        noise = torch.normal(0, self.sigma, tensor.shape)
        noise[tensor==0] = 0
        
        torch.clip(tensor+noise, 0, 1, out=tensor)
                
        return tensor  
    
    
class ManualBlur():
        
    def __init__(self, sigmas=[1,1,1]):
        
        self.eps = 10e-5
        self.sigma = sigmas
        self.r_blur = GaussianBlur(kernel_size=(11, 11), sigma=(self.sigma[0]+self.eps, self.sigma[0]+self.eps))
        self.g_blur = GaussianBlur(kernel_size=(11, 11), sigma=(self.sigma[1]+self.eps, self.sigma[1]+self.eps))
        self.b_blur = GaussianBlur(kernel_size=(11, 11), sigma=(self.sigma[2]+self.eps, self.sigma[2]+self.eps))
    
    def __call__(self, img):
        
        img[0] = self.r_blur(img[0].unsqueeze(0)).squeeze()
        img[1] = self.g_blur(img[1].unsqueeze(0)).squeeze()
        img[2] = self.b_blur(img[2].unsqueeze(0)).squeeze()

        return torch.clip(img, 0, 1)