import numpy as np
import cv2

def relight_object(normal_map, albedo_map, light_dir, mask):
    """
    Generate relit image using estimated normal map and albedo.
    
    Parameters:
        normal_map (numpy.ndarray): Estimated surface normal map (HxWx3)
        albedo_map (numpy.ndarray): Estimated albedo map (HxW)
        light_dir (numpy.ndarray): New light direction (3,)
        mask (numpy.ndarray): Binary mask of valid pixels (HxW)
    
    Returns:
        relit_img (numpy.ndarray): Image relit under the new light direction (HxW)
    """
    # TODO: Fill this functions
    # Important Notes: Consider Channels 

    # // recover (0, 1) RGB-wise normalized normal map to (-1, 1) vector-wise normalized normal map 
    normal_map = normal_map*2 - 1

    # // recover BGR-wise values to RGB-wise values
    normal_map = normal_map[..., [2, 1, 0]]  

    # // compute dot product between normal map and light direction
    ## // and clip negative values to zero (shadowed regions) 
    NL = np.tensordot(normal_map, light_dir, axes=([2], [0]))  # // (HxW)
    NL = np.maximum(NL, 0)

    # // make relit image 
    relit_img = albedo_map * NL     # // (HxW)

    # // 전체적으로 1.x 배 밝게 하기 
    relit_img = relit_img * 1.2  
    # // 범위 넘어가면 clip
    relit_img = np.clip(relit_img, 0, 1)

    # // mask the relit image 
    relit_img *= mask               # // (HxW)

    return relit_img
