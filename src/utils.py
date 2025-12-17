import numpy as np
from skimage.measure import regionprops, label
from cv2 import floodFill, bitwise_not

def postprocessing(mask):
    """
    Cleans up binary masks by:
    1. Keeping only the largest connected component.
    2. Filling holes inside the object.
    
    Parameters
    ----------
    mask : np.ndarray (batch, height, width) or (height, width)
    """
    mask = np.squeeze(mask).astype(np.uint8)

    if len(mask.shape) < 3:
        mask = np.expand_dims(mask, axis=0)
    
    n, h, w, = mask.shape
    labels_mask = np.zeros_like(mask)
    mask_out = np.zeros_like(mask)

    for i in range(n):
        # 1. Keep largest component
        labels_mask[i,] = label(mask[i,])
        regions = regionprops(labels_mask[i,])
        regions.sort(key=lambda x: x.area, reverse=True)
        
        if len(regions) > 1:
            for rg in regions[1:]:
                labels_mask[i, rg.coords[:,0], rg.coords[:,1]] = 0 
        
        labels_mask[i, labels_mask[i,] != 0] = 1
        mask[i,] = labels_mask[i,]
        
        # Normalize if necessary
        if np.max(mask[i,]) == 255:
            mask[i,] = mask[i,] / 255
            
        # 2. Fill holes (Flood Fill)
        im_flood_fill = mask[i,].copy().astype("uint8")
        overlay = np.zeros((h + 2, w + 2), np.uint8)
        floodFill(im_flood_fill, overlay, (0, 0), 255)
        im_flood_fill_inv = bitwise_not(im_flood_fill)
        
        mask_out[i,] = mask[i,] | im_flood_fill_inv
        mask_out[i,] = mask_out[i,] / 255 

    return mask_out
