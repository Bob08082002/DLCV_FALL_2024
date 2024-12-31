import os
import sys

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np


from tokenizer import BPETokenizer               
from ImageCaptionModel_v8 import ImageCaptionModel  # P2: MODIFIED ImageCaptionModel to visualize attention

def visualize_attention_from_folder(image_folder, model, tokenizer, transform, max_token=20, output_dir='./attention_maps'):
    os.makedirs(output_dir, exist_ok=True)
    # Iterate through all images in the folder
    for image_filename in os.listdir(image_folder):
        if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
    
        image_path = os.path.join(image_folder, image_filename)
        print(f"Processing {image_path}...")

        # load the image
        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        original_image = Image.open(image_path).convert("RGB")

        # get attention weights
        generated_tokens = model.autoreg_greedy(image, max_token=30)
        #generated_tokens = model.autoreg_beam_search(image, max_token=30, beam_size=3)
        decoder_blocks = model.decoder.transformer.h

        # Create an output directory for each image
        image_output_dir = os.path.join(output_dir, os.path.splitext(image_filename)[0])
        os.makedirs(image_output_dir, exist_ok=True)

        # Calculate number of rows and columns of combined image
        num_tokens = len(generated_tokens)
        num_cols = 5
        num_rows = (num_tokens + 1) // num_cols + 1  # +1 for the original image

        # Prepare subplots grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
        axes = axes.flatten()  # Flatten to make indexing easier

        # Show the original image at (0,0)
        original_image_resized = original_image.resize((224, 224))
        original_image_np = np.array(original_image_resized) / 255.0  # Normalize to [0, 1]
        axes[0].imshow(original_image_np)
        axes[0].axis('off')
        axes[0].set_title("<start>", fontsize=12)

        # Save attention maps for each token
        for token_idx, token in enumerate(generated_tokens):
            token_str = tokenizer.decode([token])
            print(token_str, end=" ")

            # get the attention weights from the last decoder block
            attn_weights = decoder_blocks[-1].attn_weights
            attn_map = attn_weights[0, :, token_idx+257, :].mean(dim=0).detach().cpu().numpy()
        
            # Take attention to image patches
            attn_map_to_image_patches = attn_map[1:257].copy()

            # Apply normalization(optional)
            att_img = (attn_map_to_image_patches - np.min(attn_map_to_image_patches)) / (np.max(attn_map_to_image_patches) - np.min(attn_map_to_image_patches))
            att_img = np.clip(att_img**2, 0, 1).reshape(16, 16)

            # interpolate to (224, 224)
            attn_map_to_image_patches_tensor = torch.tensor(att_img.reshape(16, 16)).unsqueeze(0).unsqueeze(0)
            attention_resized = F.interpolate(attn_map_to_image_patches_tensor, size=(224, 224), mode="bilinear", align_corners=False)
            attention_resized_np = attention_resized.squeeze().detach().cpu().numpy()

            # Display each attention map on the grid
            ax_idx = token_idx + 1  # Shift index by 1 since original image is at index 0
            if ax_idx < len(axes):
                axes[ax_idx].imshow(original_image_np)
                axes[ax_idx].imshow(attention_resized_np, cmap='jet', alpha=0.5)
                axes[ax_idx].axis('off')
                axes[ax_idx].set_title(token_str, fontsize=12)

        # Remove any unused subplots
        for ax in axes[num_tokens + 1:]:
            ax.axis('off')

        # Save the combined image with all tokens and attention maps
        combined_save_path = os.path.join(image_output_dir, "combined_attention.png")
        plt.tight_layout()
        plt.savefig(combined_save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
            

    print(f"Attention maps saved in {output_dir}")




if __name__ == "__main__":
    # ------------------------- Parameters -------------------------
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    # val data
    #image_folder = '../hw3_data/p3_data/images'  # path to the folder containing test images
    image_folder = sys.argv[1]  # path to the folder containing test image (ex: '../hw3_data/p3_data/images', top1 & last1 images)
    # tokenizer:
    encoder_file = "../encoder.json"
    vocab_file = "../vocab.bpe"
    tokenizer = BPETokenizer(encoder_file, vocab_file)
    # Visaul Encoder:
    vision_encoder = "vit_large_patch14_clip_224.laion2b"
    vision_encoder_size = 224
    # Text Decoder:
    text_decoder = "../hw3_data/p2_data/decoder_model.bin"  # path to the decoder weight(ex: ./decoder_model.bin)
    # Constants

    # model path
    weight_path = os.path.join("../checkpoint_model/P2/P2_model_v81_epoch7.pth")
    # output image path
    output_path = sys.argv[2] # path to the output folder

    # ------------------------- instance model and put it on device -------------------------
    model = ImageCaptionModel(vision_encoder, text_decoder)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')), strict=False)
    model = model.to(device)
    model.eval()

    # ------------------------- get validation data -------------------------
    # val transform
    val_transform = transforms.Compose([
        transforms.Resize((vision_encoder_size, vision_encoder_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std of imagenet
    ])

    # ------------------------- inference and save reult -------------------------
    visualize_attention_from_folder(image_folder, model, tokenizer, val_transform, max_token=20, output_dir=output_path)

    print("Inference Completed.")