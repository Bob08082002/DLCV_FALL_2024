# version 5: combined 2 datasets to 1 dataset + change model to deeper + T = 500

import sys

import os

import torch
import imageio as imageio
import torchvision.utils as vutils

from P1_train.modules_v5 import UNet_conditional  # use empty __init__.py s.t. UNet_conditional can be imported

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

def Sampling_specify_digit(net, alpha, alpha_hat, digit, w = 2,  T = 500, image_size = 28):
    """Given specified digit, generate 50 images of each specified class"""
    """digit is a list, ex: [0,16] means generate 50 digit 0 images of MNISTM and 50 digit 6 images of SHVN"""
    """len(digit) cant be too large, or out of GPU memory """
    """image_size is image size of model, function will return resized image with shape 28*28"""
    N = 50
    set_seed(seed = 42)
    # ========================================================== step1: x_T ~ N(0, 1) ==========================================================
    x_t = torch.randn( len(digit)*N , 3, image_size, image_size)   # Sample x_T from normal distribution with mean 0 and standard deviation 1
    x_t = x_t.to(device)                              # dtype = float32     # shape = N* C* H* W

    # --------------------- model input: y --------------------- 
    # y.shape = len(digit)*50    # y is 0 ~ 19
    tensors = [torch.full((N,), each_digit) for each_digit in digit] # Create a tensor by repeating each element 50 times
    y = torch.cat(tensors, dim=0).to(device)  # Concatenate all tensors along dim=0
     
    # ======================================================== step2: denoise for-loop =========================================================
    for t_idx in range(T-1, -1, -1): # for t_idx = T-1,...,0  # since python is 0-indexing
        # --------------------- generate z --------------------- 
        if t_idx > 0:
            z = torch.randn(len(digit)*N, 3, image_size, image_size) # Sample z from normal distribution with mean 0 and standard deviation 1
            z = z.to(device)                              
        else: # if add noise at last iteration, outcome even worse
            z = torch.zeros((len(digit)*N, 3, image_size, image_size))
            z = z.to(device)  

        # --------------------- model input: t --------------------- 
        # t.shape = len(digit)*50    # t is 0 ~ T-1
        t = torch.full((len(digit)*N, ), t_idx).to(device)  # create a tensor with all element = t_idx and it's shape = len(digit)*N

        # --------------------- Predict the noise --------------------- 
        with torch.no_grad():
           net.to(device) # put the model on gpu
           net.eval()
           # Predict the noise epsilon(x_t, t) with "conditional" generate
           epsilon_x_t_t_cond = net(x_t, t, y)  # since model and inputs are on device, epsilon_x_t_t_cond also on device

           # Predict the noise epsilon(x_t, t) with "unconditional" generate
           y = None
           epsilon_x_t_t_uncond = net(x_t, t, y)

        epsilon_x_t_t_cond = epsilon_x_t_t_cond.to(device) # 以防萬一
        epsilon_x_t_t_uncond = epsilon_x_t_t_uncond.to(device)

        # ref: Classifier-Free Diffusion Guidance(arXiv:2207.12598)
        epsilon_x_t_t = (1 + w) * epsilon_x_t_t_cond - w * epsilon_x_t_t_uncond # since epsilon_x_t_t_cond and epsilon_x_t_t_uncond are on device, epsilon_x_t_t also on device
        epsilon_x_t_t = epsilon_x_t_t.to(device) # 以防萬一

        sqrt_alpha = torch.sqrt(alpha[t_idx]).to(device) # sqrt_alpha is scalar
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t_idx]).to(device) # sqrt_one_minus_alpha_hat is scalar
        one_minus_alpha = (1 - alpha[t_idx]).to(device) # one_minus_alpha is scalar
        sqrt_one_minus_alpha = torch.sqrt(1 - alpha[t_idx]).to(device) # sqrt_one_minus_alpha is scalar

        x_t = (1/sqrt_alpha) * (x_t - (one_minus_alpha/sqrt_one_minus_alpha_hat) * epsilon_x_t_t) + sqrt_one_minus_alpha * z # x_t is on device
        x_t = x_t.to(device) # 以防萬一

    #x_t_clamped = x_t.clamp(-1, 1)                      # device same as x_t
    #x_t_normalized = (x_t_clamped + 1) / 2              # device same as x_t
    #x_t_uint8 = (x_t_normalized * 255).to(torch.uint8)  # device same as x_t

    return x_t # return n generated image ((len(digit)*50))*C*H*W) 




# do the inference & write result to output_dir
def infer_and_save(net, alpha, alpha_hat, num_each_dataset, output_image_folder):
    
    # if folder is not existed, then create it.
    if not os.path.exists(os.path.join(output_image_folder, "mnistm")):
        os.makedirs(os.path.join(output_image_folder, "mnistm"))  # Create the folder
    if not os.path.exists(os.path.join(output_image_folder, "svhn")):
        os.makedirs(os.path.join(output_image_folder, "svhn"))  # Create the folder

    """for image_label in range(num_each_dataset):
        for dataset_label in [0, 1]:
            # generate 50 conditional images for each digits
            digit = [image_label + num_each_dataset*dataset_label] 
            
            gen_images_uint8 = Sampling_specify_digit(net, alpha, alpha_hat, digit) # gen_images_uint8 is on cpu
            gen_images_uint8 = gen_images_uint8.to(torch.device("cpu")) # 以防萬一

            # Save each image with the desired name format for every i in N
            for i in range(50):
                filename = f"{image_label}_{i+1:03}.png"  # Generate filename
                if dataset_label == 0: # 0: MNISTM
                    dataset_folder = "mnistm"
                    write_image = gen_images_uint8
                else: # 1: SVHN
                    dataset_folder = "svhn"
                    write_image = gen_images_uint8

                image_path = os.path.join(output_image_folder, dataset_folder, filename)
                # Convert tensor to numpy array and save
                image = write_image[i].permute(1, 2, 0).numpy()  # Convert (C, H, W) -> (H, W, C)
                imageio.imwrite(image_path, image)
            
        print(f'50 MNISTM images and 50 SVHN images of digit {image_label} are saved.')"""

    for image_label in range(num_each_dataset):
            # generate 50 conditional images for each digits
            digit = [image_label, image_label + num_each_dataset] # ex: 6 of MNISTM & 6 of SVHN
            
            gen_images_uint8 = Sampling_specify_digit(net, alpha, alpha_hat, digit) # gen_images_uint8 is on cpu
            gen_images_uint8 = gen_images_uint8.to(torch.device("cpu")) # 以防萬一


            gen_images_uint8_MNISTM = gen_images_uint8[:50]
            gen_images_uint8_SVHN = gen_images_uint8[50:]
            # Save each image with the desired name format for every i in N
            for i in range(50):
                filename = f"{image_label}_{i+1:03}.png"  # Generate filename

                for dataset_label in [0, 1]:
                    if dataset_label == 0: # 0: MNISTM
                        dataset_folder = "mnistm"
                        write_image = gen_images_uint8_MNISTM
                    else: # 1: SVHN
                        dataset_folder = "svhn"
                        write_image = gen_images_uint8_SVHN

                    image_path = os.path.join(output_image_folder, dataset_folder, filename)

                    # Convert tensor to numpy array and save
                    #image = write_image[i].permute(1, 2, 0).numpy()  # Convert (C, H, W) -> (H, W, C)
                    #imageio.imwrite(image_path, image)

                    vutils.save_image(write_image[i], image_path, normalize=True)
            
            print(f'50 MNISTM images and 50 SVHN images of digit {image_label} are saved.')





if __name__ == '__main__':
    
    # ------------- Parameters ------------- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #!!ver4
    # class = 0 means digit 0 of MNISTM
    # class = 16 means digit 6 of SHVN 
    num_classes = 20 
    num_each_dataset = 10
    model_path = 'checkpoint_model/P1/P1_model_ver5_1.pth'
    # get $1
    output_image_folder = sys.argv[1]


    alpha = torch.load('./P1_train/alpha_tensors/alpha_tensor.pt', map_location=torch.device('cpu'))
    alpha_hat = torch.load('./P1_train/alpha_tensors/alpha_hat_tensor.pt', map_location=torch.device('cpu'))
    print(alpha.shape)
    # ------------- Load the pretrained model -------------
    net = UNet_conditional(size=28, time_dim=256, num_classes=num_classes, device=device) # !!!!!!!!!! cpu, gpu !!!!!!!!!!
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False) # !! cpu, gpu !!  #!!ver2 
    net = net.to(device)
    net.eval()

    infer_and_save(net, alpha, alpha_hat, num_each_dataset, output_image_folder)

    print("Inferece completed")