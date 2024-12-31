# version 6: combined 2 datasets to 1 dataset + change model + T = 350
# version 6_2: inference conditional & unconditional in one batch
import sys

import os

import torch
import imageio as imageio
import torchvision.utils as vutils
import torch.nn.functional as F

from P1_train.modules_v6 import ContextUnet  # use empty __init__.py s.t. UNet_conditional can be imported

def set_seed(seed):
    """Set random seed for reproducibility."""
    #random.seed(seed)               # Python random module
    #np.random.seed(seed)            # NumPy
    torch.manual_seed(seed)         # CPUE
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(seed)  # Current GPU
    #    torch.cuda.manual_seed_all(seed)  # All GPUs
    #torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    #torch.backends.cudnn.benchmark = False     # Disable autotuning

def DDPM_Sampling_specify_digit(net, T , digit, w = 2, image_size = 28):
    """Given specified digit list, generate 50 images of each digit"""
    """digit is a list, ex: [0,16] means generate 50 digit 0 images of MNISTM and 50 digit 6 images of SHVN"""
    """len(digit) cant be too large, do not exceed GPU memory """
    with torch.no_grad():
        N = 50 * len(digit)
        net.to(device) # put the model on gpu
        net.eval()
        # ========================================================== step1: x_T ~ N(0, 1) ==========================================================
        # x_t.shape = (N, C, H, W)
        x_t = torch.randn( N , 3, image_size, image_size) # Sample x_T from normal distribution with mean 0 and std 1
        x_t = x_t.to(device)                                         # dtype = float32     # shape = (N, C, H, W)

        # --------------------- model input: y --------------------- 
        # labels.shape = (2N, 20)    # y is 0 ~ 19
        y =  [torch.full((50,), each_digit) for each_digit in digit] # Create a tensor by repeating each element 50 times(concat each tensor with shape 50)
        y = torch.cat(y, dim=0)   # Concatenate all tensors along dim=0, y.shape = N
        labels_cond = F.one_hot(y, num_classes=num_classes).to(device)  # labels_cond.shape = (N, 20) 
        labels_uncond = torch.zeros_like(labels_cond)                   # labels_uncond.shape = (N, 20) 
        labels = torch.cat((labels_cond, labels_uncond), dim=0)         # labels.shape = (2N, 20)

        # ======================================================== step2: denoise for-loop =========================================================
        for t_idx in range(T-1, -1, -1): # for t_idx = T-1,...,0  # since python is 0-indexing
            # --------------------- generate z --------------------- 
            # z.shape = (N, C, H, W)
            if t_idx > 0:
                z = torch.randn((N, 3, image_size, image_size)) # Sample z from normal distribution with mean 0 and standard deviation 1                            
            else: # if add noise at last iteration, outcome even worse
                z = torch.zeros((N, 3, image_size, image_size))
            z = z.to(device)  

            # --------------------- model input: t --------------------- 
            # t.shape = 2N    # half for cond, half for uncond
            t = torch.full((2*N, ), t_idx).to(device)  # create a tensor with all element = t_idx and it's shape = 2N 

            # --------------------- model input: x_t_repeat --------------------- 
            # x_t_repeat.shape = (2N, C, H, W)
            x_t_repeat = torch.cat((x_t, x_t), dim=0)   # half for cond, half for uncond

            # --------------------- Predict the noise ---------------------       
            # Predict the noise epsilon(x_t, t) with "conditional" generate & "unconditional" generate
            if t_idx % 40 == 0:
                print(t_idx)
            noise_pred = net(x_t_repeat.float(), (t/T).float(), labels.float()) # shape = (2N, C, H, W)
            noise_pred_cond = noise_pred[:N]                                    # shape = (N, C, H, W)
            noise_pred_uncond = noise_pred[N:]                                  # shape = (N, C, H, W)

            # ref: Classifier-Free Diffusion Guidance(arXiv:2207.12598)
            # epsilon_x_t_t.shape = (N, C, H, W)
            epsilon_x_t_t = (1 + w) * noise_pred_cond - w * noise_pred_uncond # on device

            #sqrt_alpha[t_idx], sqrt_one_minus_alpha_hat[t_idx], one_minus_alpha[t_idx], sqrt_one_minus_alpha[t_idx] are scalars
            x_t = ((1/sqrt_alpha[t_idx]) * (x_t - (one_minus_alpha[t_idx]/sqrt_one_minus_alpha_hat[t_idx]) * epsilon_x_t_t) + 
                  sqrt_one_minus_alpha[t_idx] * z )# x_t is on device

    return x_t # return n generated image ((len(digit)*50))*C*H*W), 50 images for each element in digit list 








# do the inference & write result to output_dir
def infer_and_save(net, T,  num_classes, output_image_folder):
    # if folder is not existed, then create it.
    if not os.path.exists(os.path.join(output_image_folder, "mnistm")):
        os.makedirs(os.path.join(output_image_folder, "mnistm"))  # Create the folder
    if not os.path.exists(os.path.join(output_image_folder, "svhn")):
        os.makedirs(os.path.join(output_image_folder, "svhn"))  # Create the folder

    
    bs_digit = 20 # model 一次產 bs_digit * 50 images  # must num_classes % bs_digit == 0
    iter_num = int(num_classes // bs_digit) # 2

    for iter in range(iter_num): # 0 ~ 1
            # generate digit list
            # ex: iter=0: [0, 1, 2, 3],  iter=4: [16, 17, 18, 19]
            digit_list = []
            for i in range(bs_digit):
                digit_list.append(i + iter*bs_digit)
            
            # generate 50 conditional images for each digits
            gen_images_uint8 = DDPM_Sampling_specify_digit(net, T, digit_list) # gen_images_uint8 is on cpu
            gen_images_uint8 = gen_images_uint8.to(torch.device("cpu")) # 以防萬一
            # gen_images_uint8.shape = ((len(bs_digit)*50))*C*H*W)

            # Save each image with the desired name format for every i in N
            for idx, digits in enumerate(digit_list):
                for i in range(50):
                    if digits <= 9: # 0 ~ 9: MNISTM
                        dataset_folder = "mnistm"
                        filename = f"{digits}_{i+1:03}.png"  # Generate filename
                    else: # 10 ~ 19: SVHN
                        dataset_folder = "svhn"
                        filename = f"{digits-10}_{i+1:03}.png"  # Generate filename

                    image_path = os.path.join(output_image_folder, dataset_folder, filename)
                    #save tensor image
                    vutils.save_image(gen_images_uint8[idx*50 + i], image_path, normalize=True)
            
                if digits <= 9: # 0 ~ 9: MNISTM
                    print(f'50 MNISTM images of digits {digits} are saved.')
                else: # 10 ~ 19: SVHN
                    print(f'50 SVHN images of digits {digits-10} are saved.')
             







if __name__ == '__main__':
    
    # ------------- Parameters ------------- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed = 42)
    #!!ver6
    # class = 0 means digit 0 of MNISTM
    # class = 16 means digit 6 of SHVN 
    num_classes = 20 
    total_time_step = 350
    model_path = 'checkpoint_model/P1/P1_model_ver6_1.pth'
    # get $1
    output_image_folder = sys.argv[1]


    #  tensors with shape torch.Size([T]) and are on device
    alpha = torch.load('./P1_train/alpha_tensors/alpha_tensor.pt', map_location=device)
    alpha_hat = torch.load('./P1_train/alpha_tensors/alpha_hat_tensor.pt', map_location=device)
    sqrt_alpha = torch.sqrt(alpha) 
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)
    one_minus_alpha = (1 - alpha) 
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha)


    # ------------- Load the pretrained model -------------
    net = ContextUnet(in_channels=3, height=28, width=28, n_feat=128, n_cfeat=num_classes, n_downs=2) # !!!!!!!!!! cpu, gpu !!!!!!!!!!
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False) # !! cpu, gpu !!  #!!ver2 
    net = net.to(device)
    net.eval()

    infer_and_save(net=net, T=total_time_step, num_classes=num_classes, output_image_folder=output_image_folder)

    print("Inferece completed")