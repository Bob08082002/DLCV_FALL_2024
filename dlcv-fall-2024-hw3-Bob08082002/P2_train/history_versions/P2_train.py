""" ========================================= Import library ========================================= """
import os
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import  DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import loralib as lora


from CaptionDataset import CaptionDataset, collate_fn
from ImageCaptionModel_v5 import ImageCaptionModel


""" ========================================= Parameters ========================================= """
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
# training data
json_file = "../hw3_data/p2_data/train.json"
image_folder = "../hw3_data/p2_data/images/train"
# tokenizer:
encoder_file = "../encoder.json"
vocab_file = "../vocab.bpe"
# Visaul Encoder:
#vision_encoder = "vit_large_patch14_clip_224"
vision_encoder = "vit_huge_patch14_224_in21k"
vision_encoder_size = 224
# Text Decoder:
text_decoder = "../hw3_data/p2_data/decoder_model.bin"
# Constants
BATCH_SIZE = 12
START_TOKEN = 50526
END_TOKEN = 50526
PAD_TOKEN = 50526
PAD_GT_TOKEN = -100
CONSTANT = {"START_TOKEN": START_TOKEN, "END_TOKEN": END_TOKEN, "PAD_TOKEN": PAD_TOKEN, "PAD_GT_TOKEN":PAD_GT_TOKEN}
# version
version = 5

""" ========================================= Define Custom Dataset ========================================= """
# 建train transform
train_transform = transforms.Compose([
    transforms.Resize((vision_encoder_size, vision_encoder_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # mean and std of imagenet
])

# 用CaptionDataset產生資料集的train dataset，再產生trainloader
train_dataset = CaptionDataset(json_file, image_folder, encoder_file, vocab_file, train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

""" ========================================= Build Model ========================================= """
# instance model and put it on device
model = ImageCaptionModel(vision_encoder, text_decoder)
model = model.to(device)

# Setting model.require_grads (Total param should less than 10M)
# This sets requires_grad to False for all parameters without the string "lora_" in their names
#lora.mark_only_lora_as_trainable(model)
lora.mark_only_lora_as_trainable(model, bias='lora_only')
# projection_layer need to be trained
for param in model.projection_layer.parameters():
    param.requires_grad = True
print("Total params:", sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, "M")
trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad == True]

""" ========================================= Training ========================================= """
# Loss and Optimizer and epoch_n
# number of epoch
num_epochs = 20
# CE
criterion = nn.CrossEntropyLoss(ignore_index=PAD_GT_TOKEN)
# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
# Different base learning rate and update strategy #!!!!!!
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader), eta_min = 0) # update lr each batch

# Training Loop
train_loss_list = []
lr_list = []


for epoch in range(num_epochs):
    model.train() #開啟Normalize layer & DROPOUT
    train_loss = 0
    for batch_data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):  #進度條以batch完成度為趴數
        optimizer.zero_grad()
        # Prepare input data(input token and image) and ground truth data(gt token)
        padded_model_input_token = batch_data['padded_model_input_token'].to(device) # BS * K
        padded_model_gt_token = batch_data['padded_model_gt_token'].to(device) # BS * K
        images = batch_data['images'].to(device) # BS * 3 * 224 * 224

        # APPLY MODEL
        # text_prob.shape = (BS, K, 50257) 
        text_prob = model(token_id=padded_model_input_token.to(torch.int64), images=images) #token_id.shape = (BS, K), image.shape = (BS, 3, 224, 224) 

        # Update weights
        # input of CE loss should be (N, C, d1, d2, ... dm)
        # target of CE loss should be (N, d1, d2, ... dm)
        text_prob = torch.swapaxes(text_prob, 1, 2) # (BS, K, 50257) -> (BS, 50257, K) 
        loss = criterion(text_prob, padded_model_gt_token) 
        loss.backward()
        #for name, param in model.named_parameters():         ## print gradient
        #    if param.grad is not None:  # Check if gradient exists
        #        print(f"Parameter: {name} - Gradient:\n{param.grad}")
        #    else:
        #        print(f"Parameter: {name} has no gradient")

        optimizer.step()
        train_loss += loss.item()
        scheduler.step() # update lr each batch(cosine scheduler)

    
    train_loss /= len(train_loader)
    # Logging
    print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {train_loss:.4f}')
    print(f'Learning rate {scheduler.get_last_lr()[0]}')
    train_loss_list.append(train_loss)
    lr_list.append(scheduler.get_last_lr()[0])
    #scheduler.step() # update lr each epoch(linear step scheduler)

    # save the model state_dict for each epoch
    save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
    save_dir = os.path.join("../checkpoint_model", "P2", f'P2_model_v{version}_epoch{epoch}.pth')
    torch.save(save_weights, save_dir)
    print(f'Saved model  to {save_dir}')

