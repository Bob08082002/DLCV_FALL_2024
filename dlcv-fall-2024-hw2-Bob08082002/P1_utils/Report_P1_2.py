# Report problem1-2: put all 100 images in one image with column indicating different noise input,
# row indicating different digits
#
import os
from PIL import Image
import sys

def load_images_from_dir(dataset_dir, digits=10, samples_per_digit=10):
    """Loads a set number of images for each digit from the dataset directory."""
    images = {str(digit): [] for digit in range(digits)}
    
    for digit in range(digits):
        digit_str = f"{digit}_"
        digit_images = [
            os.path.join(dataset_dir, fname) 
            for fname in os.listdir(dataset_dir) 
            if fname.startswith(digit_str)
        ]
        digit_images = sorted(digit_images)[:samples_per_digit]
        images[str(digit)] = [Image.open(img_path) for img_path in digit_images]
    
    return images

def save_image_grid(images, output_path, grid_size=(10, 10), image_size=(32, 32)):
    """Saves a grid of images as a single image."""
    width, height = image_size
    grid_width, grid_height = grid_size

    # Create a blank canvas for the grid
    grid_image = Image.new('RGB', (grid_width * width, grid_height * height))

    for row in range(grid_height):
        for col in range(grid_width):
            img = images[str(row)][col]
            img = img.resize(image_size)  # Resize if necessary
            grid_image.paste(img, (col * width, row * height))

    grid_image.save(output_path)
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    img_dir = sys.argv[1]     # given image dir with generated image(ex: ~/hw2/DDPM/output_images)
    img_out_dir = sys.argv[2] # output image dir

    # if folder is not existed, then create it.
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)  # Create the folder

    # Define the dataset directories
    mnistm_dir = os.path.join(img_dir, "mnistm")
    svhn_dir = os.path.join(img_dir, "svhn")
    # Define the output large image path
    mnistm_out_path = os.path.join(img_out_dir, "mnistm_grid.png")
    svhn_out_path = os.path.join(img_out_dir, "svhn_grid.png")

    # Load images from both datasets
    mnistm_images = load_images_from_dir(mnistm_dir)
    svhn_images = load_images_from_dir(svhn_dir)

    # Save the image grids
    save_image_grid(mnistm_images, mnistm_out_path)
    save_image_grid(svhn_images, svhn_out_path)