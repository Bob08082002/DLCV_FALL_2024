# For problem 2 evaluation, given two image directory, img_dir and GT_dir. 
# it calculate MSE between images(00.png ~ 09.png) in two folders
#
from PIL import Image
import numpy as np
import os
import sys

def calculate_mse(image1, image2):
    """Calculates the Mean Squared Error between two images."""
    # Convert images to NumPy arrays
    img1_array = np.array(image1, dtype=np.float32)
    img2_array = np.array(image2, dtype=np.float32)

    # Compute the MSE
    mse = np.mean((img1_array - img2_array) ** 2)
    return mse

def mse_for_folders(folder1, folder2):
    """Calculates and prints the MSE for each pair of images in two folders."""
    folder1_files = sorted(os.listdir(folder1))
    folder2_files = sorted(os.listdir(folder2))

    # Ensure both folders have the same number of files
    if len(folder1_files) != len(folder2_files):
        raise ValueError("The two folders must contain the same number of files.")

    total_mse = 0
    for file1, file2 in zip(folder1_files, folder2_files):
        # Load the images
        img1 = Image.open(os.path.join(folder1, file1)).convert('RGB')
        img2 = Image.open(os.path.join(folder2, file2)).convert('RGB')

        # Calculate MSE for the current pair
        mse = calculate_mse(img1, img2)
        print(f"MSE for {file1} and {file2}: {mse}")

        total_mse += mse

    # Calculate average MSE across all image pairs
    avg_mse = total_mse / len(folder1_files)
    print(f"\nAverage MSE: {avg_mse}")



if __name__ == '__main__':
    img_dir = sys.argv[1]
    GT_dir = sys.argv[2]
    mse_for_folders(img_dir, GT_dir)