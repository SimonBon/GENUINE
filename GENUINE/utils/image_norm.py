import numpy as np

def normalize_image(image, mask):
    
    mask = np.invert(mask.astype(bool))
    
    image = (image / image.max()).copy()
    
    for i in range(3):
        pixels = image[mask, i].ravel()
        mean = pixels.mean()
        
        image[..., i] = np.clip(image[..., i] - mean, 0, 1)
        image[..., i] = image[..., i] / image[..., i].max()
                     
    return image
        