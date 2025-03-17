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
    rows, cols = np.where(chromeball_mask == 1)
    # TODO: Fill this functions

    for i, img_path in enumerate(tqdm(chromeballs, desc='Recover light direction')):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        # TODO: Fill this functions
        # Important Notes: You can use any opencv functions to find the brightest point!


    return L
