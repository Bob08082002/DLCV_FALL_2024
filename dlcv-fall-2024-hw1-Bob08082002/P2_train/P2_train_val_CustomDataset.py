from torch.utils.data import  Dataset
import os
from PIL import Image
import imageio.v2 as imageio
import numpy as np

class P2_CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir             # ex: ../hw1_data/p2_data/train
        self.transform = transform
        self.images = [image for image in os.listdir(data_dir) if image.endswith('_sat.jpg')]  #returns a list of filenames in data_dir which end with '_sat.jpg'(image)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """given idx, return (idx)th image and its mask in dataset"""
        img_name = self.images[idx]
        img_path = os.path.join(self.data_dir, img_name) #complete the list of filename
        mask_path = img_path.replace('_sat.jpg', '_mask.png')

        # Load image and mask
        image = imageio.imread(img_path)
        mask = imageio.imread(mask_path)
    
        # using Image.fromarray convert array(from imageio) to PIL image. PIL image之後再transform
        image = Image.fromarray(image).convert('RGB') 

        # using Image.fromarray convert array(from imageio) to PIL image.
        mask = Image.fromarray(mask).convert('RGB')
        #PIL image to numpy array
        mask = np.array(mask)
        # Define a mapping from RGB to class indices # 因為mask(GT)是rgb圖，每個pixel有以下七種顏色，分別對應到不同種類
        color_to_class = {
            (0, 255, 255): 0,  # Urban
            (255, 255, 0): 1,  # Agriculture
            (255, 0, 255): 2,  # Rangeland
            (0, 255, 0): 3,    # Forest
            (0, 0, 255): 4,    # Water
            (255, 255, 255): 5, # Barren
            (0, 0, 0): 6       # Unknown
        }
        # Convert mask to class indices
        mask_indices = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8) # create a H*W zero-value mask
        for color, index in color_to_class.items():
            mask_indices[np.all(mask == color, axis=-1)] = index
        # using Image.fromarray convert array to PIL image.
        mask = Image.fromarray(mask_indices)  # H*W mask(ie. 512*512, each pixel is an integer,which stands for its class label)
        

        # custom transform
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
    
if __name__ == '__main__':
        
    print("completed")