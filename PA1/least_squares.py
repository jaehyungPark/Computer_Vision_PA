import numpy as np
from tqdm import tqdm
import cv2

def solve_least_squares(I, L, rows, cols, mask):
    """
    Least squares solution for normal estimation and albedo recovery.
    
    Parameters:
        imgn (list of str): List of image file paths.
        I (numpy.ndarray): Intensity matrix (num_pixels, num_lights).
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

    # // LSS (1) : compute the pseudoinverse of L 
    Lt = L.T                            # // (3, 11)
    Lt_L = Lt @ L                       # // (3, 3)
    Lt_L_inv = np.linalg.inv(Lt_L)      # // (3, 3)
    Lt_L_inv_Lt = Lt_L_inv @ Lt         # // (3, 11)

    for i in tqdm(range(len(rows)), desc='Solving least squares'):
        # TODO: Fill this functions
        # Important Notes: opencv uses BGR, not RGB

        # // intensity vector for i'th pixel  
        I_i = I[i, :]                   # // (11, )

        # // LSS (2) : compute G 
        G = Lt_L_inv_Lt @ I_i           # // (3 x 11) @ (11,) -> (3, )
        
        # // get rho(albedo) and N form G 
        rho = np.linalg.norm(G, ord=2)
        if rho > 1e-5:
            N = G / rho
        else:
            N = np.zeros(3)

        # // stack to each pixel's coordinates 
        y, x = rows[i], cols[i]

        # // re-normalize the value of normal for RGB visualization : (-1, 1) -> (0, 1)
        N = (N+1)/2

        normal[y, x, :] = [N[2], N[1], N[0]]    # // opencv uses BGR, not RGB
        albedo[y, x] = rho

    # albedo = np.clip(albedo, 0, 1)              # // albedo clipping 
    # albedo = (albedo - albedo.min()) / (albedo.max() - albedo.min())    # // albedo normalization
    return normal, albedo