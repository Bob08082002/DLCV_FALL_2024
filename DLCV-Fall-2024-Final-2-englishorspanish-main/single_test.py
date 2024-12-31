import json
import os

import torch
from diffusers import DPMSolverMultistepScheduler

from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline


if __name__ == '__main__':
    pretrained_model_path = 'experiments/composed_edlora/chilloutmix/cat2_wearable_glasses_watercolor_chilloutmix/combined_model_base'
    enable_edlora = True  # True for edlora, False for lora

    pipe = EDLoRAPipeline.from_pretrained(pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler'), torch_dtype=torch.float16).to('cuda')
    with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
        new_concept_cfg = json.load(fr)
    pipe.set_new_concept_cfg(new_concept_cfg)

    #TOK = '<wearable> <glasses>'  # the TOK is the concept name when training lora/edlora
    TOK = '<watercolor1> <watercolor2>'  # the TOK is the concept name when training lora/edlora
    #prompt = f'a {TOK} in front of the door'
    prompt = 'A <watercolor1> <watercolor2> painting of a <cat21> <cat22> in a <watercolor1> <watercolor2> style wearing <wearable> <glasses>, whole body' 
    #negative_prompt = 'cropped, worst quality, low quality'
    negative_prompt = 'half body, not full body, cropped'

    image = pipe(prompt, negative_prompt=negative_prompt, height=768, width=768, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(21), guidance_scale=12).images[0]
    
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
    image.save(f'single_test.jpg')