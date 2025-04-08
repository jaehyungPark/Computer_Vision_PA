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
    # // masked area
    rows, cols = np.where(chromeball_mask == 1)
    # TODO: Fill this functions

    # // find the center and the radius of the chromeball
    x_center = np.mean(cols)
    y_center = np.mean(rows)

    x_mask_min = np.min(cols)
    x_mask_max = np.max(cols)
    radius = (x_mask_max - x_mask_min) / 2

    L = []

    for i, img_path in enumerate(tqdm(chromeballs, desc='Recover light direction')):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        # TODO: Fill this functions
        # Important Notes: You can use any opencv functions to find the brightest point!

        # // find the brightest point in the masked area of the image
        ## // apply the mask 
        masked_img = img * chromeball_mask
        ## // find how bright it is and where the point is 
        min_value, max_value, min_location, max_location = cv2.minMaxLoc(masked_img)
        x_max, y_max = max_location

        # // find the light direction based on the brightest point and the normal vector of the chromeball
        x_n = x_max - x_center
        y_n = y_max - y_center              # // 
        z_n_squared = radius**2 - x_n**2 - y_n**2

        if z_n_squared < 0:
            z_n = 0
        else:
            z_n = np.sqrt(z_n_squared)

        N = np.array([x_n, y_n, z_n])
        N = N / np.linalg.norm(N, ord=2)    # // compute L2 norm of the vector

        # // law of reflection
        R = np.array([0, 0, 1])             # // viewing direction 
        L_dir = 2 * np.dot(N, R) * N - R
        L_dir = L_dir / np.linalg.norm(L_dir, ord=2)

        L_dir[1] = - L_dir[1]               # // 

        L.append(L_dir)

    L = np.array(L)
    return L
