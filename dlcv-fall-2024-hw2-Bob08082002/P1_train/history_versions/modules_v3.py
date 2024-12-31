#-----------------------------------------------------------------#
# Reference:                                                      #
# Source: https://github.com/dome272/Diffusion-Models-pytorch     # 
# Video: https://www.youtube.com/watch?v=TBCRlnwJtZU              #
#                                                                 #
#-----------------------------------------------------------------#

# version 3: combined 2 datasets to 1 dataset + change model to smaller one(reduce down3 & up1 layer) + T = 400

import torch
import torch.nn as nn
import torch.nn.functional as F

 
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    # like self-attension in transformer
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    """2 CONV layer with gelu activation function & group norm"""
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual: # residual block
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    """MODIFIED VERSION 3: combine two dataset, which has 20 classes"""
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        
        # t.shape = N*256, while 256 is the embbeding dim "time_dim"
        # make sure time_dim == emb_dim
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels), # from 256 dim linear project to out_channels of feature map
        )

    def forward(self, x, t): # dataset_label: 0 is for MNISTM, 1 is for SVHN  # dataset_label.shape = N
        x = self.maxpool_conv(x)

        emb = torch.zeros_like(x)
        # dataset_label.shape = N
        # t.shape = N*256                                     
        # emb_layer(t).shape = N*out_channels                 
        # emb_layer(t)[:, :, None, None] = N*out_channels*1*1 
        # emb = N*out_channels*H*W, which is as same as x     
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # x.shape[-2]: H, x.shape[-1]: W

        return x + emb # add the embbeded time and output feature map


class Up(nn.Module):
    """MODIFIED VERSION 3: combine two dataset, which has 20 classes"""
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )
    def forward(self, x, skip_x, t): # dataset_label: 0 is for MNISTM, 1 is for SVHN
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1) # skip_x is from decoder, concate x and skip_x along Channel dimension
        x = self.conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) # x.shape[-2]: H, x.shape[-1]: W
        
        return x + emb



class UNet_conditional(nn.Module):
    """MODIFIED VERSION 3: combine two dataset, which has 20 classes"""
    def __init__(self, c_in=3, c_out=3, size=28, time_dim=256, num_classes=20, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64) # size
        self.down1 = Down(64, 128) # reduce size by 2, first arg is in_channel, second arg is out_channel
        self.sa1 = SelfAttention(128, int(size/2)) # size/2 
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, int(size/4)) # size/4 
        #self.down3 = Down(256, 256)
        #self.sa3 = SelfAttention(256, int(size/8)) # size/8

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 256)
        self.bot3 = DoubleConv(256, 128)


        # upsample size by 2, first arg is in_channel, second arg is out_channel
        #self.up1 = Up(512, 128)    # with corresponding shortcut connected(sa2: 256*2 = 512)
        #self.sa4 = SelfAttention(128, int(size/4))
        self.up2 = Up(256, 64)    # with corresponding shortcut connected(sa1: 128*2 = 256)
        self.sa5 = SelfAttention(64, int(size/2))
        self.up3 = Up(128, 64)    # with corresponding shortcut connected(inc: 64*2 = 128)
        self.sa6 = SelfAttention(64, int(size))
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        
        # encoding "num_classes" labels to "time_dim" long random vector, which means use random vector with length "time_dim" to represent each label
        # nn.Embedding function is similar with pos_encoding, but nn.Embedding is learnable
        self.label_emb = nn.Embedding(num_classes, time_dim) 


    def pos_encoding(self, t, channels):
        """encoding time tensor(N,1) using sine embedding, (pos_enc.shape = N*channels)"""
        """encode each of timestep into vector with length = channels"""
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
 
    def forward(self, x, t=None, y=None):
        """x=N*C*H*W,  t=N,  y=N,  dataset_label=N"""
        t = t.unsqueeze(-1).type(torch.float)   # t.shape = N*1 
        t = self.pos_encoding(t, self.time_dim) # t.shape = N*256  # encoding each of time step into vector with dim = 256
  
        # y.shape = N                 
        # label_emb(y).shape = N*256  
        # t.shape = N*256             
        if y is not None:  # if y is None, means this minibatch is unconditional training, so label info is not embedded.
            # y is labels with shape N, where each element of y is integer in range 0 ~ (num_classes-1)
            t += self.label_emb(y)
            

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        #x4 = self.down3(x3, t)
        #x4 = self.sa3(x4)

        x4 = self.bot1(x3)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        #x = self.up1(x4, x3, t) # with corresponding shortcut connected
        #x = self.sa4(x)
        x = self.up2(x4, x2, t) # with corresponding shortcut connected
        x = self.sa5(x)
        x = self.up3(x, x1, t) # with corresponding shortcut connected
        x = self.sa6(x)
        output = self.outc(x)
        return output


if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet_conditional(num_classes=20, device="cpu")
    x = torch.randn(10,3,28,28)
    t = torch.randint(0, 500, (10,))
    y = torch.randint(0, 20, (10,))
    print(net(x, t, y).shape) # torch.Size([10, 3, 32, 32])

