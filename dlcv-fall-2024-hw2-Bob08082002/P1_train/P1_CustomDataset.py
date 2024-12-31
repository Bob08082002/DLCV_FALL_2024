from torch.utils.data import  Dataset
import os
from PIL import Image
import imageio as imageio
import pandas as pd


# dataset整理image & label (from image folder & csv file)
class P1_CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
            csv_file (str): Path to the CSV file with image names and labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_frame = pd.read_csv(csv_file)  # Load the CSV file
        self.root_dir = root_dir  # Set the root directory
        self.transform = transform  # Store any transformations

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get image name and label from the data frame
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")  # Open the image as RGB #image is PIL image
        label = int(self.data_frame.iloc[idx, 1])  # Convert label to integer

        # Apply any specified transformations
        if self.transform:
            image = self.transform(image)

        return image, label