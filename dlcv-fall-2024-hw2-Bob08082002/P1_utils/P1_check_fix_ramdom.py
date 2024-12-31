# Given two folder, compare all images in them.
# To check if generations are always the same
# To check if random seed is fixed correctly
from PIL import Image
import numpy as np
import sys
import os


def compare_images(img1_path, img2_path):
    """Compares two images and returns True if they are identical, otherwise False."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Convert images to NumPy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    # Compare shapes and pixel values
    return img1_array.shape == img2_array.shape and np.array_equal(img1_array, img2_array)

def compare_all_images(folder1, folder2):
    """Compares all images from two folders and prints mismatched ones."""
    mismatched_files = []

    # Get all image filenames from both folders
    folder1_files = sorted(os.listdir(folder1))
    folder2_files = sorted(os.listdir(folder2))

    # Ensure both folders have the same number of files
    if len(folder1_files) != len(folder2_files):
        print("The folders have a different number of files.")
        return

    # Compare images with the same filenames
    for file1, file2 in zip(folder1_files, folder2_files):
        path1 = os.path.join(folder1, file1)
        path2 = os.path.join(folder2, file2)

        if not compare_images(path1, path2):
            mismatched_files.append((file1, file2))

    # Print the result
    if mismatched_files:
        print("Mismatched images:")
        for file1, file2 in mismatched_files:
            print(f"{file1} != {file2}")
    else:
        print("All images are identical.")



if __name__ == '__main__':
    # compare images in two image folder, to check if random seed is fixed
    img_dir_1 = sys.argv[1]  # given image dir 1 with generated image(ex: ~/hw2/DDPM/output_images)
    img_dir_2 = sys.argv[2]  # given image dir 2 with generated image(ex: ~/hw2/DDPM/output_images)
    
    print("---------------- MNISTM ---------------- ")
    compare_all_images(os.path.join(img_dir_1, "mnistm"), os.path.join(img_dir_2, "mnistm"))
    print("---------------- SVHN ---------------- ")
    compare_all_images(os.path.join(img_dir_1, "svhn"), os.path.join(img_dir_2, "svhn"))


    # result:
    # ---------------- MNISTM ---------------- 
    # All images are identical.
    # ---------------- SVHN ---------------- 
    # All images are identical.