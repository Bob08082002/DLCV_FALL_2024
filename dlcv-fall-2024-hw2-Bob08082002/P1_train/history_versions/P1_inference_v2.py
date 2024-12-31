import sys

import os

import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import imageio as imageio

from P1_train.modules_v2 import UNet_conditional  # use empty __init__.py s.t. UNet_conditional can be imported

def preprocess_and_resize(x_t): 
    """Given a N*3*32*32 tensor, resize it to N*3*28*28 tensor"""
    Resize_32to28 = transforms.Resize((28, 28))
    To_pil = transforms.ToPILImage()
    To_tensor = transforms.ToTensor()
    # Step 1: Clamp x to be between -1 and 1
    x_t_clamped = x_t.clamp(-1, 1)                      # device same as x_t
    # Step 2: Normalize x to be between 0 and 1
    x_t_normalized = (x_t_clamped + 1) / 2              # device same as x_t
    # Step 3: Scale to 0-255 and convert to uint8
    x_t_uint8 = (x_t_normalized * 255).to(torch.uint8)  # device same as x_t
    # Prepare a list to store resized images
    resized_images = []
    # Step 4: Loop over batch dimension (N)
    for i in range(x_t_uint8.size(0)):
        # Convert each image to PIL and resize
        pil_img = To_pil(x_t_uint8[i])                  # device: cpu(PIL doesnt support cuda)
        resized_pil_img = Resize_32to28(pil_img)        # device: cpu
        # Convert back to tensor and store
        resized_tensor = To_tensor(resized_pil_img)     # device: cpu
        resized_images.append(resized_tensor)           # device: cpu
    # Step 5: Stack tensors back into (N, C, H, W) format
    x_t_uint8_resized = torch.stack(resized_images)     # device: cpu

    return x_t_uint8_resized                            # device: cpu

def Sampling_specify_digit(net, alpha, alpha_hat, digit,  T = 400, N = 50, image_size = 32):
    """Given specified digit and dataset, generate 50(bs) images of specified class"""

    # ========================================================== step1: x_T ~ N(0, 1) ==========================================================
    x_t = torch.randn(N, 3, image_size, image_size)   # Sample x_T from normal distribution with mean 0 and standard deviation 1
    x_t = x_t.to(device)                              # dtype = float32     # shape = N* C* H* W


    # --------------------- model input: y --------------------- 
    # y.shape = N    # y is 0 ~ 19
    y = torch.full((N,), int(digit)).to(device)  # create a tensor with all element = digit and it's shape = N
    # --------------------- model input: dataset_label --------------------- 
    # dataset_label.shape = N    # dataset_label is 0 ~ 1
    #dataset_label = torch.full((N,), int(dataset_label)).to(device)  # create a tensor with all element = digit and it's shape = N
    

    # ======================================================== step2: denoise for-loop =========================================================
    for t_idx in range(T-1, -1, -1): # for t_idx = T-1,...,0  # since python is 0-indexing
        if t_idx > 0:
            z = torch.randn(N, 3, image_size, image_size) # Sample z from normal distribution with mean 0 and standard deviation 1
            z = z.to(device)                              
        else: # if add noise at last iteration, outcome even worse
            z = torch.zeros((N, 3, image_size, image_size))
            z = z.to(device)  

        # --------------------- model input: t --------------------- 
        # t.shape = N    # t is 0 ~ T-1
        t = torch.full((N,), t_idx).to(device)  # create a tensor with all element = t_idx and it's shape = N


        with torch.no_grad():
           net.to(device) # put the model on gpu
           net.eval()
           # Predict the noise epsilon(x_t, t) with "conditional" generate
           epsilon_x_t_t_cond = net(x_t, t, y)  # since model and inputs are on device, epsilon_x_t_t_cond also on device
        epsilon_x_t_t_cond = epsilon_x_t_t_cond.to(device) # 以防萬一

        sqrt_alpha = torch.sqrt(alpha[t_idx]).to(device) # sqrt_alpha is scalar
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t_idx]).to(device) # sqrt_one_minus_alpha_hat is scalar
        one_minus_alpha = (1 - alpha[t_idx]).to(device) # one_minus_alpha is scalar
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha[t_idx]).to(device) # sqrt_one_minus_alpha is scalar

        x_t = (1/sqrt_alpha) * (x_t - (one_minus_alpha/sqrt_one_minus_alpha_hat) * epsilon_x_t_t_cond) + sqrt_one_minus_alpha * z # x_t is on device
        x_t = x_t.to(device) # 以防萬一

    return x_t # return n generated image (N*C*H*W) 


        



# do the inference & write result to output_dir
def infer_and_save(net, alpha, alpha_hat, num_classes, output_image_folder):

    # if folder is not existed, then create it.
    if not os.path.exists(os.path.join(output_image_folder, "mnistm")):
        os.makedirs(os.path.join(output_image_folder, "mnistm"))  # Create the folder
    if not os.path.exists(os.path.join(output_image_folder, "svhn")):
        os.makedirs(os.path.join(output_image_folder, "svhn"))  # Create the folder

    for dataset_label in [0, 1]:  # 0: MNISTM, 1: SVHN
        for image_label in range(10):
            # generate 50 conditional images for each digits
            # Given specified digit and dataset, generate 50 images of specified class
            # gen_images.shape = 50*3*28*28
            digit = image_label + 10 * dataset_label #!!ver2
            gen_images_uint8 = Sampling_specify_digit(net, alpha, alpha_hat, digit=digit) # gen_images_uint8 is on cpu
            gen_images_uint8 = gen_images_uint8.to(torch.device("cpu")) # 以防萬一


            # Save each image with the desired name format for every i in N
            for i in range(50):
                filename = f"{image_label}_{i+1:03}.png"  # Generate filename

                if dataset_label == 0: # 0: MNISTM
                    dataset_folder = "mnistm"
                else: # 1: SVHN
                    dataset_folder = "svhn"

                image_path = os.path.join(output_image_folder, dataset_folder, filename)
                # Convert tensor to numpy array and save
                image = gen_images_uint8[i].permute(1, 2, 0).numpy()  # Convert (C, H, W) -> (H, W, C)
                imageio.imwrite(image_path, image)







if __name__ == '__main__':
    
    # ------------- Parameters ------------- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 20 #!!ver2
    #device = torch.device("cpu")
    # get $1
    output_image_folder = sys.argv[1]
    alpha = torch.load('./P1_train/alpha_tensors/alpha_tensor.pt', map_location='cpu')
    alpha_hat = torch.load('./P1_train/alpha_tensors/alpha_hat_tensor.pt', map_location='cpu')



    # ------------- Load the pretrained model -------------
    net = UNet_conditional(size=32, time_dim=256, num_classes=num_classes, device=device) # !!!!!!!!!! cpu, gpu !!!!!!!!!!
    net.load_state_dict(torch.load('checkpoint_model/P1/P1_model_ver2_1.pth', map_location="cpu"), strict=False) # !! cpu, gpu !!  #!!ver2 
    net = net.to(device)
    net.eval()

    infer_and_save(net, alpha, alpha_hat, num_classes, output_image_folder)


    #x_t_uint8 = Sampling_specify_digit(net, alpha, alpha_hat, 0, 6, w = 3,  T = 1000, N = 10, image_size = 32)
    #torch.save(x_t_uint8, './P1_temp_output/x_t_uint8_2.pt')
    #save_tensor_images(x_t_uint8, "./P1_temp_output")
    #x_t_uint8 = torch.load('./P1_temp_output/x_t_uint8.pt', map_location='cpu')
    #print(x_t_uint8)


    print("Inferece completed")