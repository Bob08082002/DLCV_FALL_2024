# This utils.py is for Problem 2
import torch
import numpy as np
def beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # 1. beta
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    # 2. alpha
    alpha = 1 - betas
    # 3. alpha_hat
    # alpha hat = [alpha[0], alpha[0]*alpha[1], alpha[0]*alpha[1]*alpha[2], ... ,alpha[0]*...*alpha[T-1])] 
    alpha_hat = torch.zeros_like(alpha) # tensor with len T, alpha_hat.shape = T
    alpha_hat[0] = alpha[0] # Set the first element
    for i in range(1, alpha.shape[0]):
        alpha_hat[i] = alpha_hat[i - 1] * alpha[i]

    return betas, alpha, alpha_hat

# 999, ... 0 with linear spaced 50 elements
# set initial_time = 1 for better MSE
def uniform_timestep_scheduler(total_timesteps=1000, num_steps=50, initial_time = 1): 
    equal_interval = total_timesteps // num_steps #20
    time = initial_time # must < 20
    timestep_list = []
    while time < total_timesteps:
        timestep_list.append(time)
        time += equal_interval

    reversed_timestep_list = timestep_list[::-1]
    return   reversed_timestep_list


