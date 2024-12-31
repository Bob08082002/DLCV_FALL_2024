# Report_P2_1: for report Prob2-1: generate 00.pt ~ 03.pt with different eta in one grid
# eta = {0, 0.25, 0.5, 0.75, 1.0}
#
import sys

import os
import numpy as np
import torch
import imageio as imageio
from utils import *
from UNet import UNet
import torchvision.utils as vutils


def set_seed(seed):
    """Set random seed for reproducibility."""
    #random.seed(seed)               # Python random module
    #np.random.seed(seed)            # NumPy
    torch.manual_seed(seed)         # CPU
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(seed)  # Current GPU
    #    torch.cuda.manual_seed_all(seed)  # All GPUs
    #torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    #torch.backends.cudnn.benchmark = False     # Disable autotuning

def DDIM_Sampling(net, alpha, alpha_hat, eta, timesteps, noise_xT):

    # ========================================================== step1: x_T ~ N(0, 1) ==========================================================
    # provided by TA. ex: 00.pt ~ 09.pt
    x_t = noise_xT
    x_t = x_t.to(device)  # dtype = float32   
    
    # ======================================================== step2: denoise for-loop =========================================================
    for i, t_idx in enumerate(timesteps):
        # --------------------- model input: t --------------------- 
        # t.shape = bs = 1    # t is T-1,...,0
        t = torch.tensor([t_idx], device=device) # shape = torch.Size([1]), if without [], shape = torch.Size([]) 

        # --------------------- Predict the noise --------------------- 
        with torch.no_grad():
           net.to(device) # put the model on gpu
           net.eval()
           # Predict the noise epsilon(x_t, t)
           predicted_noise = net(x_t, t)  # since model and inputs are on device, predicted_noise also on device
        predicted_noise = predicted_noise.to(device) # 以防萬一

        # --------------------- alpha_t and alpha_prev_t ---------------------  
        # !! alpha in DDIM paper = alpha_hat in DDPM paper(different notation) !!
        alpha_t = alpha_hat[t_idx]
        if t_idx >= 20: # equal_interval = 20
            ## !! alpha(t-1)不是t_idx-1，而是到下一個timestep
            # ex: 50 time steps = 999, 979, 959 ... , 0. If current t = 979, then t - 1 should be 959, not 978.
            alpha_prev_t = alpha_hat[timesteps[i+1]]  
        else: 
            alpha_prev_t = alpha_hat[0]   # define alpha[-1] = 1 (ref: DDIM paper)
        alpha_t = alpha_t.to(device) # 以防萬一
        alpha_prev_t = alpha_prev_t.to(device) # 以防萬一

        #  --------------------- Compute the variance term σ(η) --------------------- 
        sigma_t = eta * torch.sqrt((1 - alpha_prev_t) / (1 - alpha_t) * (1 - alpha_t / alpha_prev_t))
        sigma_t = sigma_t.to(device)# 以防萬一

        # --------------------- predicted x_0 --------------------- 
        x0_t = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        x0_t = torch.clamp(x0_t, min=-1.0, max=1.0) # clip predicted x0: better MSE
        x0_t = x0_t.to(device) # 以防萬一

        # --------------------- compute the direction pointing to x[t-1] --------------------- 
        direction = torch.sqrt(1 - alpha_prev_t - sigma_t**2) * predicted_noise
        direction = direction.to(device) # 以防萬一

        # --------------------- random_noise --------------------- 
        random_noise = sigma_t * torch.randn_like(noise_xT)
        random_noise = random_noise.to(device) # 以防萬一

        # --------------------- update x_t for the next step --------------------- 
        x_t = torch.sqrt(alpha_prev_t) * x0_t + direction + random_noise
        x_t = x_t.to(device) # 以防萬一

    return x_t # return n generated image


# Min-max normalization function
def min_max_normalize(tensor):
    min_val = tensor.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]  # (B, 1, 1, 1)
    max_val = tensor.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]  # (B, 1, 1, 1)
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-9)  # Avoid division by zero
    return normalized_tensor

# do the inference
def infer_and_save(net, alpha, alpha_hat, eta, timesteps, input_noise_folder):

    # load input noise, ex: 00.pt ~ 03.pt
    input_noise_list = []
    for i in range(4):
        noise_path = os.path.join(input_noise_folder, f"0{i}.pt")
        input_noise = torch.load(noise_path, map_location=device)
        input_noise_list.append(input_noise)

    gen_image_list = []
    for i, input_noise in enumerate(input_noise_list):
            noise_xT = input_noise.to(device)
            gen_images = DDIM_Sampling(net, alpha, alpha_hat, eta, timesteps, noise_xT) # gen_images is on cpu
            gen_images = min_max_normalize(gen_images)
            gen_images = gen_images.to(torch.device("cpu")) # 以防萬一
            gen_image_list.append(gen_images)

    return gen_image_list





if __name__ == '__main__':
    set_seed(seed = 42)
    # ------------- Parameters ------------- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get $1 $2 $3
    input_noise_folder = sys.argv[1]
    output_image_folder = sys.argv[2] # dontcare
    pretrain_weight_path = sys.argv[3]
    #initial_time = sys.argv[4]
    #initial_time = int(initial_time)

    total_timesteps=1000
    num_steps=50
    initial_time = 1 # for initial_time = 0:10, initial_time = 1 has best MSE

    # beta, alpha, alpha_hat are tensors
    betas, alpha, alpha_hat = beta_scheduler(n_timestep=total_timesteps, linear_start=1e-4, linear_end=2e-2) # shape = 1000
    timesteps = uniform_timestep_scheduler(total_timesteps=total_timesteps, num_steps=num_steps, initial_time=initial_time)

    # ------------- Load the pretrained model -------------
    net = UNet() #net is on cpu
    net.load_state_dict(torch.load(pretrain_weight_path, map_location=torch.device('cpu')), strict=False)
    net = net.to(device)
    net.eval()


    eta = 0.0
    gen_image_list_eta000 = infer_and_save(net, alpha, alpha_hat, eta, timesteps, input_noise_folder)
    eta = 0.25
    gen_image_list_eta025 = infer_and_save(net, alpha, alpha_hat, eta, timesteps, input_noise_folder)
    eta = 0.50
    gen_image_list_eta050 = infer_and_save(net, alpha, alpha_hat, eta, timesteps, input_noise_folder)
    eta = 0.75
    gen_image_list_eta075 = infer_and_save(net, alpha, alpha_hat, eta, timesteps, input_noise_folder)
    eta = 1.0
    gen_image_list_eta100 = infer_and_save(net, alpha, alpha_hat, eta, timesteps, input_noise_folder)

    # Concatenate the lists along a new dimension to create a (20, 3, H, W) tensor
    all_images = torch.cat(gen_image_list_eta000 + gen_image_list_eta025 + gen_image_list_eta050 + gen_image_list_eta075 + gen_image_list_eta100, dim=0)  # Shape: (20, 3, 28, 28)

    # Save the grid image
    image_path = "./Report_P2_1_different_eta/grid_image.png"
    vutils.save_image(all_images, image_path, nrow=4)

    print(f"Grid image saved to {image_path}")

    print("Inferece completed")