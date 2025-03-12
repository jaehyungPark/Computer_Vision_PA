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
        NormalMap (numpy.ndarray): Estimated surface normals.
        Albedo (numpy.ndarray): Estimated albedo.
    """
    #TODO: Fill this functions
    
    return NormalMap, Albedo