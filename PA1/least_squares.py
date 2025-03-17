import numpy as np
from tqdm import tqdm
import cv2

def solve_least_squares(I, L, rows, cols, mask):
    """
    Least squares solution for normal estimation and albedo recovery.
    
    Parameters:
        imgn (list of str): List of image file paths.
        L (numpy.ndarray): Light directions (11, 3).
        rows (numpy.ndarray): Row indices of the object pixels.
        cols (numpy.ndarray): Column indices of the object pixels.

    Returns:
        normal (numpy.ndarray): Estimated surface normals.
        albedo (numpy.ndarray): Estimated albedo.
    """
    h, w = mask.shape
    normal = np.zeros((h, w, 3))
    albedo = np.zeros((h, w))
    # TODO: Fill this functions

    for i in tqdm(range(len(rows)), desc='Solving least squares'):
        # TODO: Fill this functions
        # Important Notes: opencv uses BGR, not RGB
    
    return normal, albedo