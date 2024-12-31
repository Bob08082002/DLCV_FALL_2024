import os
import argparse
import json

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm, trange
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from omegaconf import OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument( ## $2
    "--outdir", 
    type=str,
    nargs="?",
    help="dir to write results to",
    default="P3_2_multiple_concept"
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=22,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=10,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--config",
    type=str,
    default="../stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument( ## $3
    "--ckpt",
    type=str,
    #default="models/ldm/stable-diffusion-v1/model.ckpt", 
    default="../stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument( ##  $1
    "--jsonfile",
    type=str,
    default='../hw2_data/textual_inversion/input.json',
    help="path to input.json",
)
parser.add_argument( 
    "--trained_embeddings_dog",
    type=str,
    default="../checkpoint_model/P3/trained_embedding/trained_embeddings_dog_epoch60.pt",
    help="path to trained token embedding of 'dog'",
)
parser.add_argument( 
    "--trained_embeddings_david_revoy",
    type=str,
    default="../checkpoint_model/P3/trained_embedding/trained_embeddings_david_revoy_epoch140.pt",
    help="path to trained token embedding of 'David Revoy'",
)
opt = parser.parse_args()

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


# (ref: txt2img.py)
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model



if __name__ == "__main__": 
    set_seed(seed = 142)
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # batch_size 
    BATCH_SIZE = 1 # must be 1, or out of CUDA memory
    # get $1 $2 $3
    jsonfile_path = opt.jsonfile
    output_image_folder = opt.outdir
    pretrain_model_path = opt.ckpt


    # ============================================ READ INPUT.JSON ============================================
    # Read input.json
    f = open(jsonfile_path)
    inputjson = json.load(f)
    # get token name
    token_names = [inputjson["0"]["token_name"], inputjson["1"]["token_name"]] # token_names of concept 0 & 1

    # Extract the prompt lists for all concepts
    #all_prompts = [concept["prompt"] for concept in inputjson.values()]
    #dog_prompts_list = all_prompts[0]          # len can be more than 2
    #david_revoy_prompts_list = all_prompts[1]  # len can be more than 2
    #prompts_list = [dog_prompts_list, david_revoy_prompts_list]

    # ============================================ LOAD MODEL ============================================
    # Load the model(ref: txt2img.py) & the pretrain weight using config
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{pretrain_model_path}")
    model = model.to(device)
    sampler = DPMSolverSampler(model)
    # Add pseudo-tokens special tokens: <new1> and <new2> and modify token_embeddings (49408, 768) -> (49410, 768)
    model.cond_stage_model.tokenizer.add_tokens(token_names)
    model.cond_stage_model.transformer.resize_token_embeddings(len(model.cond_stage_model.tokenizer))
    #print(len(model.cond_stage_model.tokenizer)) #49410

    # Load the saved embeddings
    loaded_embeddings_dog = torch.load(opt.trained_embeddings_dog, map_location=device)
    loaded_embeddings_david_revoy= torch.load(opt.trained_embeddings_david_revoy, map_location=device)
    # Replace the corresponding embeddings in the model's input embedding layer
    with torch.no_grad():
        model.cond_stage_model.transformer.get_input_embeddings().weight[-2] = loaded_embeddings_dog
        model.cond_stage_model.transformer.get_input_embeddings().weight[-1] = loaded_embeddings_david_revoy

    # ============================================ INFERENCE LOOP ============================================
    data = [BATCH_SIZE * [f"a {token_names[0]} in the style of {token_names[1]}"]] # one prompt, ex:"a <new1> in the style of <new2>"
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([BATCH_SIZE, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
    with torch.no_grad():
        with model.ema_scope():
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.get_learned_conditioning(BATCH_SIZE * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=BATCH_SIZE,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta,
                                                    x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)


                    for x_sample in x_checked_image_torch:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        img = Image.fromarray(x_sample.astype(np.uint8))
                        img.save(os.path.join(output_image_folder, f"multiple_concept_{n}.png"))