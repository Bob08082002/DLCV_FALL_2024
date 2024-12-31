import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from tokenizer import BPETokenizer



# define custom dataset
class CaptionDataset(Dataset):
    def __init__(self, json_file, image_folder, encoder_file, vocab_file, transform=None,
                 CONSTANT={"START_TOKEN": 50256, "END_TOKEN": 50256, "PAD_TOKEN": 50256, "PAD_GT_TOKEN":-100}):
        # load constants
        self.START_TOKEN = CONSTANT["START_TOKEN"]
        self.END_TOKEN = CONSTANT["END_TOKEN"]
        self.PAD_TOKEN = CONSTANT["PAD_TOKEN"]
        self.PAD_GT_TOKEN = CONSTANT["PAD_GT_TOKEN"]

        # read json file
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.annotations = data['annotations']
        self.images = {img['id']: img['file_name'] for img in data['images']} # dicts with image id and it's file name
        
        # Initialize tokenizer
        self.tokenizer = BPETokenizer(encoder_file, vocab_file)
        
        # Folder containing images
        self.image_folder = image_folder
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.annotations) # len = #caption. If one image has many captions, it creates multiple data pairs.

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        caption = annotation['caption']
        image_id = annotation['image_id']
        image_path = os.path.join(self.image_folder, self.images[image_id])
        
        # Process the image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize the caption
        context = self.tokenizer.encode(caption)
        model_input_token = [self.START_TOKEN] + context + [self.END_TOKEN] # [start_token, gt1, gt2, ... gtN, end_token] 
        gt_token = context + [self.END_TOKEN] # [gt1, gt2, ... gtN, end_token]
        
        return {
            'model_input_token': torch.tensor(model_input_token),
            'gt_token': torch.tensor(gt_token),
            'image': image
        }


# Padding function for variable-length tokens
# since each caption has different legth N, 
# we need to pad each caption to same length K within a batch, 
# where K is max length of model_input_token in that batch.
def collate_fn(batch, CONSTANT={"START_TOKEN": 50256, "END_TOKEN": 50256, "PAD_TOKEN": 50256, "PAD_GT_TOKEN":-100}):
    # load constants
    START_TOKEN = CONSTANT["START_TOKEN"]
    END_TOKEN = CONSTANT["END_TOKEN"]
    PAD_TOKEN = CONSTANT["PAD_TOKEN"]
    PAD_GT_TOKEN = CONSTANT["PAD_GT_TOKEN"]

    # Get max length of model_input_token in this batch
    max_len = max(len(item['model_input_token']) for item in batch)
    
    padded_model_input_tokens = []
    padded_gt_tokens = []
    images = []
    
    for item in batch:
        # Pad model_input_token to max_len
        model_input_token = item['model_input_token']
        # ex:  padded_input_token = [start_token, gt1, gt2, ... gtN, end_token, PAD_TOKEN, PAD_TOKEN, PAD_TOKEN]
        # total length = max length of model_input_token in this batch = K
        padded_input_token = torch.cat([model_input_token, torch.tensor([PAD_TOKEN] * (max_len - len(model_input_token)))])
        
        # Pad gt_token to max_len
        gt_token = item['gt_token']
        # ex:  padded_gt_token = [gt1, gt2, ... gtN, end_token, PAD_GT_TOKEN, PAD_GT_TOKEN, PAD_GT_TOKEN, PAD_GT_TOKEN]
        # total length = max length of model_input_token in this batch = K
        padded_gt_token = torch.cat([gt_token, torch.tensor([PAD_GT_TOKEN] * (max_len - len(gt_token)))])
        

        # Collect image and padded tokens
        images.append(item['image'])
        padded_model_input_tokens.append(padded_input_token)
        padded_gt_tokens.append(padded_gt_token)
    
    return {
        'padded_model_input_token': torch.stack(padded_model_input_tokens),
        'padded_model_gt_token': torch.stack(padded_gt_tokens),
        'images': torch.stack(images)
    }