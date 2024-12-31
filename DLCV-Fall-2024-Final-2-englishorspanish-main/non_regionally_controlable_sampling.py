import json
import os
import argparse
import torch
from datetime import datetime
from diffusers import DPMSolverMultistepScheduler

from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline

def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--pretrained_model', default='experiments/composed_edlora/anythingv4/hina+kario+tezuka+mitsuha+son_anythingv4/combined_model_base', type=str)
    parser.add_argument('--sketch_condition', default=None, type=str)
    parser.add_argument('--sketch_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_sketch_adaptor_weight', default='', type=str)
    parser.add_argument('--keypose_condition', default=None, type=str)
    parser.add_argument('--keypose_adaptor_weight', default=1.0, type=float)
    parser.add_argument('--region_keypose_adaptor_weight', default='', type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--prompt', default='photo of a toy', type=str)
    parser.add_argument('--negative_prompt', default='', type=str)
    parser.add_argument('--prompt_rewrite', default='', type=str)
    parser.add_argument('--seed', default=16141, type=int)
    parser.add_argument('--suffix', default='', type=str)
    parser.add_argument('--num_image', default=1, type=int)
    parser.add_argument('--image_height', default=512, type=int)
    parser.add_argument('--image_width', default=512, type=int)
    parser.add_argument('--token', default='', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pretrained_model_path = args.pretrained_model
    enable_edlora = True  # True for edlora, False for lora

    pipe = EDLoRAPipeline.from_pretrained(pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler'), torch_dtype=torch.float16).to('cuda')
    with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
        new_concept_cfg = json.load(fr)
    pipe.set_new_concept_cfg(new_concept_cfg)

    prompt = args.prompt
    #negative_prompt = 'cropped, worst quality, low quality'
    negative_prompt = args.negative_prompt

    # save_dir = os.path.join(args.save_dir, f'generated_images') # use this when not inference 

    # if os.path.exists(save_dir):
    #     # Create a timestamped folder name
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     save_dir = f"{save_dir}_{timestamp}"
    # else:
    #     # Use the base folder name if it doesn't exist
    #     save_dir = save_dir

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    num_images = args.num_image  # Number of images to generate

    for i in range(num_images):
        seed = i + args.seed
        image = pipe(prompt, negative_prompt=negative_prompt, height=args.image_height, width=args.image_width, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(seed), guidance_scale=15).images[0]
        final_W, final_H = image.size
        #compare which is bigger
        if final_H >= final_W:
            if final_H >= 512:
                ratio = final_H/512
                final_H = 512
                final_W = final_W / ratio
        else:
            if final_W >= 512:
                ratio = final_W/512
                final_W = 512
                final_H = final_H / ratio

        image=image.resize((int(final_W),int(final_H)))

        save_name = f'{i}.png'
        
        save_path = os.path.join(save_dir, save_name)
        image.save(save_path)