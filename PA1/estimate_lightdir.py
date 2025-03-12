import numpy as np

def estimate_light_direction(N, I, mask):
    """
    Estimate the light direction L from the Normal Map (N) and Intensity (I), considering only valid pixels from the mask.
    
    Parameters:
        N (numpy.ndarray): Normal map (h, w, 3), where each pixel is a 3D vector.
        I (numpy.ndarray): Intensity values (h, w) for unknown image.
        mask (numpy.ndarray): Binary mask of valid pixels (h, w).
    
    Returns:
        L (numpy.ndarray): Estimated light direction (3, ) for unknown image.
    """
    #TODO: Fill this functions
    
    return L