import sys
import os
sys.path.append(os.path.join(os.getcwd(),'P2_train')) # to import tokenizer and ImageCaptionModel in P2_train
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from P2_train.tokenizer import BPETokenizer
from P2_train.ImageCaptionModel_v8 import ImageCaptionModel

 
class INFERENCE_CaptionDataset(Dataset):
    """ do not need tokenizer in inference stage """
    def __init__(self, image_folder, transform=None):  
        self.image_folder = image_folder
        self.transform = transform if transform else transforms.ToTensor()

        # Read all image files in the folder
        self.image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        image_path = os.path.join(self.image_folder, file_name)
        # Load and transform the image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return {
            'image': image,
            'file_name': file_name
        }
    

def generate_captions(model, tokenizer, val_loader, json_output_path):
    model.eval()  
    output_dict = {}
    
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch["image"].to(device)  # move images to device, shape (BS, 3, 224, 224), BS=1 in inference
            file_names = batch["file_name"]

            # Generate captions using the model's auto-regressive method
            #generated_tokens = model.autoreg_beam_search(images, max_token=20, beam_size=3)  # beam search (list on CPU)
            generated_tokens = model.autoreg_greedy(images, max_token=30)  # greedy search (list on CPU)

            # Remove tokens after the END_TOKEN if present
            if END_TOKEN in generated_tokens:
                end_idx = generated_tokens.index(END_TOKEN)
                generated_tokens = generated_tokens[:end_idx]  # Truncate at END_TOKEN

            # Convert token IDs to text using the tokenizer
            caption = tokenizer.decode(generated_tokens)

            # Store the result in the output dictionary
            for file_name in file_names:
                base_name = os.path.splitext(file_name)[0]  # Remove the extension (.jpg, .png, etc.)
                output_dict[base_name] = caption
                print(f"{base_name}: {caption}")

    # Save the output dictionary to a JSON file
    with open(json_output_path, 'w') as json_file:
        json.dump(output_dict, json_file, indent=4)
    print(f"Captions saved to {json_output_path}")


if __name__ == "__main__":
    # ------------------------- Parameters -------------------------
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    # val data
    image_folder = sys.argv[1]  # path to the folder containing test images
    # tokenizer:
    encoder_file = "./encoder.json"
    vocab_file = "./vocab.bpe"
    tokenizer = BPETokenizer(encoder_file, vocab_file)
    # Visaul Encoder:
    vision_encoder = "vit_large_patch14_clip_224.laion2b"
    vision_encoder_size = 224
    # Text Decoder:
    text_decoder = sys.argv[3]  # path to the decoder weight(ex: ./decoder_model.bin)
    # Constants
    INFERENCE_BATCH_SIZE = 1  # must be 1 during inference
    START_TOKEN = 50256
    END_TOKEN = 50256
    PAD_TOKEN = 50256
    PAD_GT_TOKEN = -100
    CONSTANT = {"START_TOKEN": START_TOKEN, "END_TOKEN": END_TOKEN, "PAD_TOKEN": PAD_TOKEN, "PAD_GT_TOKEN":PAD_GT_TOKEN}
    # model path 
    weight_path = os.path.join("./checkpoint_model/P2/P2_model_v81_epoch7.pth")
    # output path
    json_output_path = sys.argv[2]  # path to the output json file

    # ------------------------- instance model and put it on device -------------------------
    model = ImageCaptionModel(vision_encoder, text_decoder)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')), strict=False)
    model = model.to(device)
    model.eval()

    # ------------------------- get validation data -------------------------
    # val transform
    val_transform = transforms.Compose([
        transforms.Resize((vision_encoder_size, vision_encoder_size), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std of imagenet
    ])

    # 用INFERENCE_CaptionDataset產生資料集的val dataset，再產生val_loader
    val_dataset = INFERENCE_CaptionDataset(image_folder, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=INFERENCE_BATCH_SIZE, shuffle=False)

    # ------------------------- inference and save reult -------------------------
    generate_captions(model, tokenizer, val_loader, json_output_path)

    print("Inference Completed.")

