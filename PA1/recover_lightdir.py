import numpy as np
from tqdm import tqdm
import cv2

def recover_light_direction(chromeballs, chromeball_mask):
    """
    Recovers the light direction from images of chromeballs by detecting the brightest spot on each ball.
    
    Parameters:
        chromeballs (list of str): List of file paths to chromeball images.
        chromeball_mask (np.ndarray): Binary mask indicating the region of the chromeball in the image.

    Returns:
        L (np.ndarray): Array of recovered light directions (N x 3), where N is the number of chromeball images.
    """
    #TODO: Fill this functions
    
    return L
