import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from recover_lightdir import recover_light_direction
from least_squares import solve_least_squares
from rpca import ialm
from relight import relight_object

def args_parser():
    parser = argparse.ArgumentParser(description='PA1: Photometric stereo')
    parser.add_argument('-d', '--dataset_root', type=str, default='./PA1_dataset')
    parser.add_argument('-o', '--object', type=str, default='all')
    parser.add_argument('-i', '--image_cnt', type=int, default=11)

    args = parser.parse_args()
    return args

def compute_mse(original, estimated):
    return np.mean((original - estimated) ** 2)

def main():
    #############################################################################
    ############################## Step0: Settings ##############################
    #############################################################################
    # Parse command-line arguments
    args = args_parser()
    if args.object=='all':
        objects = ['choonsik', 'toothless', 'nike', 'moai']

    else:
        objects = args.object
        
    # Make output dirs
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    #############################################################################
    #################### Step1: Recovery for Light Direction ####################
    #############################################################################
    if os.path.isfile(f"{output_dir}/light_dirs.npy"):
        light_dirs = np.load("./output/light_dirs.npy")
    else:    
        chromeballs = [f"{args.dataset_root}/chromeball/bmp/{i}.bmp" for i in range(args.image_cnt)]
        chromeball_mask = cv2.imread(f"{args.dataset_root}/chromeball/mask/mask.bmp", cv2.IMREAD_GRAYSCALE) / 255.0
            
        # Recover Light Directions
        light_dirs = recover_light_direction(chromeballs, chromeball_mask)
        np.save(f"{output_dir}/light_dirs.npy", light_dirs)
    print(light_dirs)
    
    for obj in objects:
        print(f"processing {obj}")
        obj_dir = os.path.join(output_dir, obj)
        os.makedirs(obj_dir, exist_ok=True)
        
        input_dir = os.path.join(args.dataset_root, obj)
        
        # Load Data
        images = [f"{input_dir}/bmp/{i}.bmp" for i in range(args.image_cnt)]
        image_mask = cv2.imread(f"{input_dir}/mask/mask.bmp", cv2.IMREAD_GRAYSCALE) / 255.0

        # Solve Least Squares for Normal Map
        rows, cols = np.where(image_mask == 1)
        print(rows.shape)

        I = np.zeros((len(rows), args.image_cnt))

        for i, img_path in enumerate(tqdm(images, desc='Load images')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
            for j in range(len(rows)):
                I[j, i] = img[rows[j], cols[j]]

        normal_map, albedo = solve_least_squares(I, light_dirs.T, rows, cols, image_mask)

        os.makedirs(f"{output_dir}/{obj}/ls", exist_ok=True)
        cv2.imwrite(f"{output_dir}/{obj}/ls/normal_map.png", (normal_map * 255).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{obj}/ls/albedo_map.png", (albedo * 255 * image_mask).astype(np.uint8))
        
        # MC (IALM)
        A_hat, E_hat, iter = ialm(I)
        normal_map, albedo = solve_least_squares(A_hat, light_dirs.T, rows, cols, image_mask)

        os.makedirs(f"{output_dir}/{obj}/rpca", exist_ok=True)
        cv2.imwrite(f"{output_dir}/{obj}/rpca/normal_map.png", (normal_map * 255).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{obj}/rpca/albed_map.png", (albedo * 255 * image_mask).astype(np.uint8))
        
        # Estimate Light Direction for Unknown Light Condition
        unknown_image = cv2.imread(f"{input_dir}/bmp/unknown.bmp", cv2.IMREAD_GRAYSCALE) / 255.0
        cv2.imwrite(f"{output_dir}/{obj}/unknown_image.png", (unknown_image * 255).astype(np.uint8))

        unknown_light_dir = np.load("./input/unknown_light_dir.npy")
        
        # Relighting
        relit_image = relight_object(normal_map, albedo, unknown_light_dir, image_mask)
        cv2.imwrite(f"{output_dir}/{obj}/relit_image.png", (relit_image * 255).astype(np.uint8))
        
        # Compute MSE
        mse = compute_mse(unknown_image, relit_image)
        with open(f"{obj_dir}/mse.txt", "w") as f:
            f.write(f"mse: {mse}\n")
        print(f"{obj}: mse = {mse}")

if __name__ == "__main__":
    main()