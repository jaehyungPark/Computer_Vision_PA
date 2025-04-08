import numpy as np
import time

def shrink(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)

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
        A (np.ndarray): The recovered low-rank component. (of I = albedo*NL)
        E (np.ndarray): The recovered sparse error component.
        iterations (int): The number of iterations executed.
    """
    m, n = D.shape
    iterations = 0
    Y = np.zeros((m, n))
    A = np.zeros((m, n))
    E = np.zeros((m, n))

    # TODO: Fill this functions
    # Important Notes: You can use any functions in numpy to implement svd!
    # Recommend you to use frobenius norm in numpy

    # // hyperparameter 
    lamb = 1 / np.sqrt(m)           # // sparsity weight 
    mu = 1                          # // penalty parameter for augmented Lagrangian
    rho = 1.1                       # // increasing ratio of mu
    mu_bar = mu * 1e7               # // upper bound of mu
    
    err = np.inf
    print("initial err: ", err)

    iterations = 0
    start_time = time.time()

    while err > threshold:

        # // update A : singular value thresholding
        U, S, Vt = np.linalg.svd((1/mu)*Y + D - E, full_matrices=False)
        S_shrink = shrink(S, 1/mu)
        A = U @ np.diag(S_shrink) @ Vt

        # // update E 
        E = shrink(D - A + (1/mu)*Y, lamb/mu)

        # // update Y and mu 
        Y = Y + mu*(D - A - E)
        mu = min(rho*mu, mu_bar)

        # // check if D converged
        err = np.linalg.norm(D - A - E, 'fro') / np.linalg.norm(D, 'fro')

        iterations += 1
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for IALM to converge: {elapsed_time:.4f} sec")

    return A, E, iterations
