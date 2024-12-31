# Modified decoder:
# Apply Lora, and modified decoder to generate caption based on both text and visual embeddings 
#
import math
import collections
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import loralib as lora

class Config:

    def __init__(self, checkpoint=None):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.lora_rank = 32

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=cfg.lora_rank) # Apply Lora
        self.c_proj = lora.Linear(cfg.n_embd, cfg.n_embd, r=cfg.lora_rank)     # Apply Lora
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=cfg.lora_rank)),  # Apply Lora
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=cfg.lora_rank)) # Apply Lora
        ]))

    def forward(self, text_img_embed): 
        text_embed = text_img_embed[0] # text_embed shape = (BS, K, 768)
        img_embed = text_img_embed[1] # img_embed shape = (BS, 257, 768)

        # concat text_embed and img_embed along dim=1
        x = torch.cat((img_embed, text_embed), dim=1)  # x shape = (BS, K+257, 768)
        # pass concated text_embed and img_embed to self-attetion and MLP
        x = x + self.attn(self.ln_1(x)) # x shape = (BS, K+257, 768)
        x = x + self.mlp(self.ln_2(x))  # x shape = (BS, K+257, 768)

        # only use text part of x as next Block's text_embed input. # ver3 & ver1
        # all Block use same img_embed from visual encoder.
        # x shape = (BS, K+257, 768)
        K = text_embed.shape[1] # K is max length of model input token in this batch
        text_embed = x[:, -K:, :]  # text_embed.shape = (BS, K, 768)
        
        return (text_embed, img_embed)

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = lora.Embedding(cfg.vocab_size, cfg.n_embd, r=cfg.lora_rank), # Apply Lora
            wpe = lora.Embedding(cfg.block_size, cfg.n_embd, r=cfg.lora_rank), # Apply Lora
            h = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = lora.Linear(cfg.n_embd, cfg.vocab_size, bias=False, r=cfg.lora_rank) # Apply Lora
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

    def forward(self, token_id: Tensor, img_embed: Tensor):
        """ token_id.shape = (BS, K), img_embed.shape = (BS, 257, 768)"""
        token_id = torch.narrow(token_id, 1, 0, min(token_id.size(1), self.block_size))
        pos = torch.arange(token_id.size()[1], dtype=torch.long, device=token_id.device).unsqueeze(0)
        #text_embed.shape=torch.Size([BS, K, 768])
        text_embed = self.transformer.wte(token_id) + self.transformer.wpe(pos)  


        # text_embed.shape = (BS, K, 768)
        (text_embed, img_embed) = self.transformer.h((text_embed, img_embed))
        text_part = self.lm_head(self.transformer.ln_f(text_embed)) # x.shape = (BS, K, 50257)
        return text_part
