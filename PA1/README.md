# PA1: Photometric Stereo
This repository contains the code for a programming assignment on photometric stereo. The goal of this assignment is to recover the surface normals and albedo of an object from multiple images captured under varying lighting conditions, and then to use this information to relight the object under an unknown lighting condition.

- Due: Apr.5
- TA session: Apr.1 & Apr.3
- TA: Wooseok Jeon (jws5271a@gm.gist.ac.kr)

**NO PLARIARISM, NO DELAY, DON'T USE AI SUPPORTER (If you do not comply, you will be given an F.)**

## Overview

The provided code follows these main steps:
1. **Recover Light Directions:**  
   Use images of a chromeball to compute the light directions.  
   *Function to implement:* `recover_light_direction`

2. **Compute Normal Map and Albedo (Least Squares):**  
   For each object, extract pixel intensities under different lighting conditions and solve a least squares problem to estimate the surface normal map and albedo.  
   *Function to implement:* `solve_least_squares`

3. **Robust Principal Component Analysis (RPCA):**  
   Apply the Iterative Augmented Lagrangian Multiplier (IALM) method to decompose the intensity matrix into low-rank and sparse components for improved robustness.  
   *Function to implement:* `lagrangian`

4. **Estimate Light Direction:**  
   Estimate the light direction for an image captured under an unknown lighting condition using the computed normal map.  
   *Function to implement:* `estimate_light_direction`

5. **Relighting:**  
   Generate a relit image of the object using the estimated light direction, the normal map, and the albedo.  
   *Function to implement:* `relight_object`

## Directory Structure
You can download the PA1_dataset [here]().
```
. 
├── PA1_dataset/            # Input dataset directory 
│ ├── chromeball/           # Chromeball images and mask for light direction recovery
│ ├── choonsik/             # Object images and mask 
│ ├── toothless/ 
│ ├── nike/ 
│ └── moai/ 
├── output/                 # Output directory (created automatically) to save results 
├── main.py                 # Main Python script (the provided code) 
├── recover_lightdir.py     # Module to implement recover_light_direction 
├── least_squares.py        # Module to implement solve_least_squares 
├── ialm.py                 # Module to implement lagrangian (RPCA) 
├── relight.py              # Module to implement relight_object 
└── estimate_lightdir.py    # Module to implement estimate_light_direction
```

## Requirements

- **Python Version:** 3.6 or above
- **Libraries:**  
  - OpenCV (`cv2`)
  - NumPy
  - argparse
  - tqdm

You can install the required libraries using pip:

```
pip install opencv-python numpy tqdm
```

## Usage
Run the main script from the command line. The script accepts the following

arguments:

- `-d` or `--dataset_root`: Path to the dataset directory (default: ./PA1_dataset)
- `-o` or `--object`: Object to process (default: all). If set to a specific object name (e.g., choonsik), only that object will be processed.
- `-i` or `--image_cnt`: Number of images to use (default: 11)


## Implementation Details
The script performs the following tasks:

1. Light Direction Recovery:
   - Loads chromeball images and the corresponding mask.
   - Calls recover_light_direction to compute the light directions.
   - Saves the computed light directions to output/light_dirs.npy.

2. Processing Each Object:
   - For each specified object (or all objects if all is selected), the script:
   - Loads the object’s images and mask.
   - Extracts pixel intensities from the images where the mask is active.
   - Calls solve_least_squares with the intensity matrix and recovered light directions to compute the normal map and albedo.
   - Saves the resulting normal map and albedo images under the object's output directory.

3. RPCA Using IALM:
   - Decomposes the intensity matrix using the lagrangian function.
   - Recomputes the normal map and albedo from the low-rank approximation.
   - Saves the RPCA results.

4. Unknown Lighting and Relighting:
   - Loads an image with an unknown lighting condition.
   - Uses estimate_light_direction to determine the light direction.
   - Calls relight_object to produce a relit image of the object.
   - Computes the Mean Squared Error (MSE) between the unknown image and the relit image.
   - Saves the relit image and the MSE value.

## Functions to Implement

For this assignment, you are required to implement the following functions/modules:
   
- recover_light_direction(chromeballs, chromeball_mask)

    Recover the light directions from the chromeball images.

- solve_least_squares(I, light_dirs_T, rows, cols, image_mask)

    Solve a least squares problem to obtain the normal map and albedo from the image intensities.

- lagrangian(I)

    Apply the Iterative Augmented Lagrangian Multiplier (IALM) method for matrix decomposition (RPCA).

- estimate_light_direction(normal_map, unknown_image, image_mask)

    Estimate the light direction from the unknown lighting condition image.

- relight_object(normal_map, albedo, estimated_light_dir, image_mask)

    Generate a relit image of the object using the computed normal map, albedo, and the estimated light direction.

## Output
The script generates the following outputs for each object:

- Normal Map: Saved as normal_map.png under both ls (least squares) and rpca (RPCA) directories.
- Albedo Map: Saved as albedo.png under both ls and rpca directories.
- Unknown Image: Saved as unknown_image.png for reference.
- Relit Image: Saved as relit_image.png.
- MSE Value: A text file (mse.txt) containing the mean squared error between the unknown image and the relit image.

All outputs are saved in the output/ directory.