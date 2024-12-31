# Report_P2_2: for report Prob2-2: generate images using slerp and lerp between 00.pt and 01.pt
# alpha = {0.0, 0.1, ..., 1.0}
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

    # Set alpha values for interpolation
    alpha_values = torch.tensor(np.linspace(0, 1, 11))  # 0, 0.1, 0.2, ..., 1.0

    # load input noise, ex: 00.pt ~ 01.pt
    input_noise_list = []
    for alpha_value in alpha_values:  # 0, 0.1, 0.2, ..., 1.0
            noise_path_00 = os.path.join(input_noise_folder, f"00.pt")
            input_noise_00 = torch.load(noise_path_00, map_location=device)
            noise_path_01 = os.path.join(input_noise_folder, f"01.pt")
            input_noise_01 = torch.load(noise_path_01, map_location=device)

            noise_slerp = slerp(alpha_value, input_noise_00, input_noise_01)
            noise_lerp = lerp(alpha_value, input_noise_00, input_noise_01)

            input_noise_list.append([noise_slerp, noise_lerp])

    gen_image_list_slerp = []
    gen_image_list_lerp = []
    # inference with 11 slerp noise and 11 lerp noise
    for i, input_noise in enumerate(input_noise_list): # 0, 0.1, 0.2, ..., 1.0
            noise_slerp, noise_lerp = input_noise
            noise_slerp = noise_slerp.to(device)
            noise_lerp = noise_lerp.to(device)

            gen_images_slerp = DDIM_Sampling(net, alpha, alpha_hat, eta, timesteps, noise_slerp) # gen_images is tensor on device
            gen_images_slerp = min_max_normalize(gen_images_slerp)
            gen_images_slerp = gen_images_slerp.to(torch.device("cpu")) # 以防萬一

            gen_images_lerp = DDIM_Sampling(net, alpha, alpha_hat, eta, timesteps, noise_lerp) # gen_images is tensor on device
            gen_images_lerp = min_max_normalize(gen_images_lerp)
            gen_images_lerp = gen_images_lerp.to(torch.device("cpu")) # 以防萬一

            
            gen_image_list_slerp.append(gen_images_slerp)
            gen_image_list_lerp.append(gen_images_lerp)

    return gen_image_list_slerp, gen_image_list_lerp


def slerp(t, v0, v1):
    """Spherical Linear Interpolation (Slerp) between two vectors."""
    # Calculate the dot product
    dot_product = torch.clamp(torch.dot(v0.view(-1), v1.view(-1))/(v0.norm(p=2) * v1.norm(p=2)), -1.0, 1.0)
    theta = torch.acos(dot_product)  # angle between v0 and v1
    sin_theta = torch.sin(theta)
    if sin_theta < 1e-6:
        print(sin_theta)
        return (1 - t) * v0 + t * v1  # Linear interpolation if angle is very small
    # Perform Slerp
    return (torch.sin((1 - t) * theta) / sin_theta) * v0 + (torch.sin(t * theta) / sin_theta) * v1

def lerp(t, v0, v1):
    """Linear Interpolation (Lerp) between two vectors."""
    return (1 - t) * v0 + t * v1


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
    gen_image_list_slerp, gen_image_list_lerp = infer_and_save(net, alpha, alpha_hat, eta, timesteps, input_noise_folder)


    # Convert lists to tensors
    slerp_images = torch.cat(gen_image_list_slerp, dim=0)  # Shape: (11, 3, 28, 28)
    lerp_images = torch.cat(gen_image_list_lerp, dim=0)  # Shape: (11, 3, 28, 28)
    

    # Save the grid images
    vutils.save_image(lerp_images, './Report_P2_2_slerp_lerp/lerp_grid.png', nrow=11)
    vutils.save_image(slerp_images, './Report_P2_2_slerp_lerp/slerp_grid.png', nrow=11)

    print("Grid images saved: 'lerp_grid.png' and 'slerp_grid.png'")


    print("Inferece completed")