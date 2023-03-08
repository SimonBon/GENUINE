import os
import cv2
import numpy as np

from tqdm                   import tqdm
from tifffile               import imread
from .device                 import best_gpu
from cellpose.models        import CellposeModel
from utils.image_norm      import normalize_image
from skimage.segmentation   import expand_labels

def get_matching_files(base_dir, match=None):
    
    files_list = []
    for root, _, files in os.walk(base_dir):
        for name in files:
            if ".tif" in name.lower():
                
                if isinstance(match, (list, str)):
                    
                    if isinstance(match, str):
                        match = [match]
                      
                    files_list.append(os.path.join(root, name)) if any([m.lower() in name.lower() for m in match]) else None
                    
                else:
                    
                    files_list.append(os.path.join(root, name))
                    
    return files_list
                    

def segment_images(image_dir, sample, n_files, model_kwargs, diameter):

    files_list = get_matching_files(image_dir, sample)
                                            
    np.random.shuffle(files_list)
    files_list = files_list[:n_files]
    
    print(files_list)
        
    images = [imread(x) for x in files_list]
    dapis = [im[..., 2] for im in images]

    model = CellposeModel(**model_kwargs, device=best_gpu())
    masks, _, _ = model.eval(dapis, diameter, channels=[[0, 0]]*len(images), normalize=True)
    
    for m in masks:
        m = expand_labels(m, distance=4)
        
    return masks, images


def Images2H5(image_dir, model_kwargs, sample, diameter=70, n_files=10, patch_sz=64, ret_masks=False):
    
    masks, images = segment_images(image_dir, sample, n_files, model_kwargs, diameter)
    
    patch_sz = int(patch_sz)
    
    return_patches = []
    return_masks = []
    for mask, image in zip(masks, images):
        
        normed = normalize_image(image, mask)
        centers = get_extractable_cells(mask, patch_sz)
        if ret_masks:
            patches, masks = extract_patches(centers, normed, mask, patch_sz, ret_masks=ret_masks)
            return_patches.extend(patches)
            return_masks.extend(masks)
        else:
            patches = extract_patches(centers, normed, mask, patch_sz, ret_masks=ret_masks)
            return_patches.extend(patches)

    if ret_masks:
        return (np.array(return_patches), np.array(return_masks))
    
    else:
        return np.array(return_patches)
        
        
    
def extract_patches(centers, image, mask, patch_sz, ret_masks):
    
    excluded = 0
    patches = []
    masks = []
    for c in centers:
        
        patch = image[c[0]-patch_sz:c[0]+patch_sz, c[1]-patch_sz:c[1]+patch_sz].copy()
        mask_patch = mask[c[0]-patch_sz:c[0]+patch_sz, c[1]-patch_sz:c[1]+patch_sz].copy()
        mask_patch[mask_patch != c[2]] = 0
        idxs = np.where(mask_patch)
        
        if np.any(idxs[0] == 0) or np.any(idxs[1] == 0) or np.any(idxs[0]==(2*patch_sz)-1) or np.any(idxs[0]==(2*patch_sz)-1):
            excluded += 1
            continue
        
        background = np.invert(mask_patch.astype(bool))
    
        for i in range(3):
            
            if patch[..., i].max() == 0:
                excluded += 1
                continue
            
            patch[..., i] = patch[..., i] / patch[..., i].max()
            
        patch[background] = 0
        patches.append(patch)
        
        if ret_masks:
            masks.append(mask_patch)
        
    print("excluded: ", excluded)

    if ret_masks:
        return patches, masks
    else:
        return patches

def get_extractable_cells(mask: np.ndarray, patch_sz) -> np.ndarray:
    
    mask_values = np.unique(mask)
    mask_values = np.delete(mask_values, np.where(mask_values == 0)[0])
    
    centers = []  
    for n in tqdm(mask_values):
        
        tmp = np.copy(mask)
        tmp[mask != n] = 0
        tmp = tmp.astype(float)
        y, x = np.where(tmp != 0)
        y_min, y_max = y.min(), y.max()+1
        x_min, x_max = x.min(), x.max()+1
        
        if y_min <= patch_sz or y_max >= mask.shape[0]-patch_sz or x_min<=patch_sz or x_max>=mask.shape[1]-patch_sz:
            continue
        
        cell = tmp[y_min:y_max, x_min:x_max]
        y, x = calc_center(cell)
        y_center, x_center = np.round(y+y_min,0), np.round(x+x_min,0)
        centers.append([y_center, x_center, n])
        
    return np.array(centers).astype(np.uint16)
        
def calc_center(bin):
    
    M = cv2.moments(bin)
    return M["m10"]/M["m00"], M["m01"]/M["m00"]