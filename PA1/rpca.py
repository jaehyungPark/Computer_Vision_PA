import numpy as np

def shrink(X, tau):
    """
    Applies the shrinkage (soft-thresholding) operator element-wise on X.
    For each element in X, it computes:
        sign(X) * max(|X| - tau, 0)
    This is commonly used in sparse coding and L1 minimization.
    
    Parameters:
        X (np.ndarray): Input array.
        tau (float): Threshold parameter.
        
    Returns:
        result (np.ndarray): The result after applying the shrinkage operator.
    """
    #TODO: Fill this functions
    
    return result

def ialm(D, threshold=1e-3):
    """
    Performs low-rank and sparse decomposition of the input matrix D using
    the Inexact Augmented Lagrange Multiplier (IALM) method.
    
    The algorithm decomposes D into:
        D = A + E
    where A is a low-rank matrix and E is a sparse error matrix.
    
    Parameters:
        D (np.ndarray): The input data matrix of shape (m, n).
        threshold (float): Convergence threshold based on the residual change.
        
    Returns:
        A (np.ndarray): The recovered low-rank component.
        E (np.ndarray): The recovered sparse error component.
        iterations (int): The number of iterations executed.
    """
    #TODO: Fill this functions
    
    return A, E, iterations
