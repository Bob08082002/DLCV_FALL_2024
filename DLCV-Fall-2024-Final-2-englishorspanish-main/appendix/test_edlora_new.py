import json,os,copy

import torch
from diffusers import DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline
from typing import List,Dict,Any,Tuple

from mixofshow_new.pipelines.pipeline_edlora import EDLoRAPipeline
import torch
from mixofshow_new.pipelines.pipeline_edlora import EDLoRAPipeline, StableDiffusionPipeline,bind_concept_prompt
from mixofshow_new.utils.convert_edlora_to_diffusers import convert_edlora, load_new_concept, merge_lora_into_weight
import argparse
import os
import json

from transformers.models.clip.tokenization_clip import CLIPTokenizer
from transformers.models.clip.modeling_clip import CLIPTextModel
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoder_kl import AutoencoderKL

def print_keys_recursively(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{prefix}{key}")
            print_keys_recursively(value, prefix=prefix + "  ")
#a = torch.load('output/task_1/models/edlora_model-latest.pth')
#print_keys_recursively(a)
"""
# Structure of a pt
    params:
        new_concept_embedding # Nums of imgs:
            <TOK1>: _embedding1_
            <TOK2>: _embedding2_
            ...
        text_encoder
            text_model.encoder.layers.<0~11>.self_attn.<q/k/v/out>_proj.lora_<up/down>.weight: _weight_ #96
        unet
            down_blocks.<0/1/2>.attentions.<0/1>.transformer_blocks.0.attn<1/2>.to_<q/k/v/out.0>.lora_<down/up>.weight: _weight_ #96
            up_blocks.<1/2/3>.attentions.<0/1/2>.transformer_blocks.0.attn<1/2>.to_<q/k/v/out>.lora_<down/up>.weight: _weight_ #144
            mid_block.attentions.0.transformer_blocks.0.attn<1/2>.to_<q/k/v/out>.lora_<down/up>.weight: _weight_ #16

"""
##############################
########    FINISHED   #######
##############################
def parse_new_concepts(concept_cfg):
    with open(concept_cfg, 'r') as f:
        concept_list = json.load(f)

    model_paths = [concept['lora_path'] for concept in concept_list]
    embedding_list = []
    #text_encoder_list = []
    unet_list = []

    for model_path in model_paths:
        model = torch.load(model_path)['params']

        if 'new_concept_embedding' in model and len(
                model['new_concept_embedding']) != 0:
            embedding_list.append(model['new_concept_embedding'])
        else:
            print(f'Warning: No "new_concept_embedding" is found in {model_path}')
            embedding_list.append(None)

        if 'text_encoder' in model and len(model['text_encoder']) != 0:
            print(f'Warning:"text_encoder" lora weights is found in {model_path}')
            print(f'This is deprecated due to CIFC adopt a text-encoder frozen training strategy')
            #text_encoder_list.append(model['text_encoder'])
        #else:
            #text_encoder_list.append(None)

        if 'unet' in model and len(model['unet']) != 0:
            unet_list.append(model['unet'])
        else:
            print(f'Warning: No "unet" is found in {model_path}')
            unet_list.append(None)
            
    embedding_list: List[Dict[str,torch.Tensor]]
    unet_list:List[Dict[str,torch.Tensor]]
    concept_list:List[Dict[str,Any]]
    
    return embedding_list, unet_list, concept_list

def merge_new_concepts_(embedding_list:List[Dict[str,torch.Tensor]], 
                        concept_list:Dict[Any,Any] ,
                        tokenizer:CLIPTokenizer,
                        text_encoder:CLIPTextModel):
    def add_new_concept(concept_name, embedding):
        new_token_names = [
            f'<new{start_idx + layer_id}>'
            for layer_id in range(NUM_CROSS_ATTENTION_LAYERS)
        ]
        num_added_tokens = tokenizer.add_tokens(new_token_names)
        assert num_added_tokens == NUM_CROSS_ATTENTION_LAYERS
        new_token_ids = [
            tokenizer.convert_tokens_to_ids(token_name)
            for token_name in new_token_names
        ]

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        token_embeds[new_token_ids] = token_embeds[new_token_ids].copy_(
            embedding[concept_name])

        embedding_features.update({concept_name: embedding[concept_name]})

        return start_idx + NUM_CROSS_ATTENTION_LAYERS, new_token_ids, new_token_names

    embedding_features = {}
    new_concept_cfg = {}

    start_idx = 0

    NUM_CROSS_ATTENTION_LAYERS = 16

    for idx, (embedding,concept) in enumerate(zip(embedding_list, concept_list)):
        concept_names = concept['concept_name'].split(' ')

        for concept_name in concept_names:
            if not concept_name.startswith('<'):
                continue
            else:
                assert concept_name in embedding, 'check the config, the provide concept name is not in the lora model'
            start_idx, new_token_ids, new_token_names = add_new_concept(concept_name, embedding)
            new_concept_cfg.update({
                concept_name: {
                    'concept_token_ids': new_token_ids,
                    'concept_token_names': new_token_names
                }
            })
    return embedding_features, new_concept_cfg

@torch.no_grad()
def compute_weight_to_combine_lora(
    tokenizer:CLIPTokenizer, text_encoder:CLIPTextModel,
    prompt:str, all_concept_cfg:Dict[str,Dict[str,List[Any]]],
    all_concept_emb: Dict[str,torch.Tensor], concept_list:List[Dict[str,Any]] 
        )->torch.Tensor:
    
    NUM_OF_ATTENTION_LAYERS = 16
    DEVICE =  text_encoder.device
    print(f'There are {len(concept_list)} LoRA to be merged...')
    
    def prompt2layerWisePrompts(prompt:str,all_concept_cfg:Dict[str,Dict[str,List[Any]]])->List[str]:
        layer_prompts = [prompt]*NUM_OF_ATTENTION_LAYERS
        
        # Replace each concept token in global prompt to formulate layer-wise prompts
        for concept_token,_v in all_concept_cfg.items():
            layer_wise_tokens  =  _v["concept_token_names"]
            assert len(layer_wise_tokens) == len(layer_prompts), 'len(layer_wise_tokens) != len(layer_prompts)'
            if concept_token in prompt:
                layer_prompts = [layer_prompt.replace(concept_token,single_layer_token) 
                                    for (layer_prompt,single_layer_token) in zip(layer_prompts,layer_wise_tokens)]
            #else:
                #print(f'{concept_token} is not in {prompt}')
                
        return layer_prompts
    
    def prompts2embs(layer_wise_prompts:List[str],tokenizer:CLIPTokenizer,text_encoder:CLIPTextModel)->torch.Tensor:
        token_ids:torch.Tensor = tokenizer(layer_wise_prompts, return_tensors="pt", padding="longest", truncation=True)["input_ids"].to(DEVICE) # (L, Np)
        layer_wise_embs:torch.Tensor = text_encoder(token_ids)["last_hidden_state"].to(torch.float)  # Shape: (L, Np, D)
        return layer_wise_embs

    def layerWiseConceptEmbs(concept_list:List[Dict[str,Any]],
                             all_concept_emb:Dict[str,torch.Tensor])->torch.Tensor:
                
        all_concept_avg_embs = []
        for concept in concept_list:
            concept_embs = [all_concept_emb[concept_token].clone() for concept_token in concept['concept_name'].split(' ')]
            all_concept_avg_embs.append(torch.mean(torch.stack(concept_embs), dim=0))
           
        return torch.stack(all_concept_avg_embs)
            
    layer_wise_prompts = prompt2layerWisePrompts(prompt,all_concept_cfg) # prompts of size(L, Np)
    layer_wise_prompt_embs = prompts2embs(layer_wise_prompts,tokenizer,text_encoder) #(L, Np, D)
    layer_wise_all_concept_embs = layerWiseConceptEmbs(concept_list,all_concept_emb).to(DEVICE) #(G, L, D)
    
    L, Np, D = layer_wise_prompt_embs.shape
    G, _, _ = layer_wise_all_concept_embs.shape
    
    # Initialize the result tensor
    layer_scores = torch.zeros((L, G), device=layer_wise_prompt_embs.device)
    
    for l in range(L):
        # Extract embeddings for the current layer
        _prompt_embs = layer_wise_prompt_embs[l]  # Shape: (Np, D)
        _concept_embs = layer_wise_all_concept_embs[:, l, :]  # Shape: (G, D)
        
        dot_product = torch.matmul(_prompt_embs, _concept_embs.T)  #(Np, D) @ (D, G) -> (Np, G)
        max_scores = torch.max(dot_product, dim=0).values # Get max along Np axis: (Np, G) -> (G,)
        layer_scores[l] = max_scores/max_scores.sum() # Normalize 
    
    return layer_scores # (L,G)

@torch.no_grad()
def merge_unet(unet:UNet2DConditionModel,
               weight:torch.Tensor,
               unet_list:List[Dict[str,torch.Tensor]])->UNet2DConditionModel:
    # Sanity check
    _ALL_KEYS = unet_list[0].keys()
    for i in range(1,len(unet_list)):
        assert unet_list[i].keys() == _ALL_KEYS, 'Not all unet_lora have same keys'
    
    # This is the order of layers initialized in "EDLoRAPipeline"
    _LAYERNAME_IDX_DICT = {
        'down_blocks.0.attentions.0.':0,
        'down_blocks.0.attentions.1.':1,
        'down_blocks.1.attentions.0.':2,
        'down_blocks.1.attentions.1.':3,
        'down_blocks.2.attentions.0.':4,
        'down_blocks.2.attentions.1.':5,
        'mid_block.attentions.0.':6,
        'up_blocks.1.attentions.0.':7,
        'up_blocks.1.attentions.1.':8,
        'up_blocks.1.attentions.2.':9,
        'up_blocks.2.attentions.0.':10,
        'up_blocks.2.attentions.1.':11,
        'up_blocks.2.attentions.2.':12,
        'up_blocks.3.attentions.0.':13,
        'up_blocks.3.attentions.1.':14,
        'up_blocks.3.attentions.2.':15,
    }
    
    print("Start merging weights...")
    unet_state_dict = unet.state_dict()
    DEVICE = unet.device

    # Collect the weight with same names and use correct coefficient for weighted sum
    for name in unet_list[0].keys():
        assert ("lora_down.weight" in name) or ("lora_up.weight" in name),\
            f'No lora_up or lora_down in your lora layer "{name}"'
        name:str
        
        if "lora_down.weight" in name:
            base_name = name.replace("lora_down.","")
            up_name = name.replace("lora_down.","lora_up.")
            which_layer = _LAYERNAME_IDX_DICT[name.split('transformer_blocks')[0]]
            
            # Get lora_up and lora_down and lora_up * lora_down
            down_weights = [unet[name] for unet in unet_list]
            up_weigts = [unet[up_name] for unet in unet_list]
            lora_weights = [torch.matmul(u, d) for (u,d) in zip(up_weigts,down_weights)]
            coef = weight[which_layer]
            assert len(coef) == len(lora_weights),'len(coef) != len(lora_weights)'

            # Compute weighted sum and them add back to original weight
            _stacked_tensors = torch.stack(lora_weights, dim=0).to(DEVICE)
            _coef = coef.view(-1, *([1] * (_stacked_tensors.dim() - 1))).to(DEVICE)

            unet_state_dict[base_name] = unet_state_dict[base_name]+torch.sum(_stacked_tensors * _coef, dim=0)
            
    unet.load_state_dict(unet_state_dict)
    print("Finish merging weights...")
    return unet
    

##############################
######## TODO : MODIFY #######
##############################


##############################
#########  ARCHIVED   ########
##############################

TEMPLATE_SIMPLE = 'photo of a {}'
def get_hooker(module_name):
    def hook(module, feature_in, feature_out):
        if module_name not in module_io_recoder:
            module_io_recoder[module_name] = {'input': [], 'output': []}
        if record_feature:
            module_io_recoder[module_name]['input'].append(feature_in[0].cpu())
            if module.bias is not None:
                if len(feature_out.shape) == 4:
                    bias = module.bias.unsqueeze(-1).unsqueeze(-1)
                else:
                    bias = module.bias
                module_io_recoder[module_name]['output'].append(
                    (feature_out - bias).cpu())  # remove bias
            else:
                module_io_recoder[module_name]['output'].append(
                    feature_out.cpu())

    return hook

@torch.no_grad()
def get_text_feature(prompts, tokenizer, text_encoder, device, return_type='category_embedding'):
    text_features = []

    if return_type == 'category_embedding':
        for text in prompts:
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_length=True,
                return_overflowing_tokens=False,
                padding='do_not_pad',
            ).input_ids

            new_token_position = torch.where(torch.tensor(tokens) >= 49407)[0]
            # >40497 not include end token | >=40497 include end token
            concept_feature = text_encoder(
                torch.LongTensor(tokens).reshape(
                    1, -1).to(device))[0][:,
                              new_token_position].reshape(-1, 768)
            text_features.append(concept_feature)
        return torch.cat(text_features, 0).float()
    elif return_type == 'full_embedding':
        text_input = tokenizer(prompts,
                               padding='max_length',
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors='pt')
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
        return text_embeddings
    else:
        raise NotImplementedError


##############################
######## MAIN FUNCTION #######
##############################
def cifc():
    my_test_prompt =  'a <cat21> <cat22> '

    # NOTE: Load pipeline structure of SD1.5
    # NOTE: "tokenizer" and "text_encoder" are modified here

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    concept_cfg = 'datasets/data_cfgs/MixofShow/multi-concept/real/cat2_wearable_glasses_watercolor_chilloutmix.json' # Modified according to task

    # Initialize SD1.5 pipeline
    model_id = 'experiments/pretrained_models/chilloutmix'
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
    train_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder='scheduler') # Not used
    test_scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder='scheduler')
    pipe.safety_checker = None
    pipe.scheduler = test_scheduler
    tokenizer:CLIPTokenizer = pipe.tokenizer
    text_encoder:CLIPTextModel = pipe.text_encoder
    unet:UNet2DConditionModel = pipe.unet
    vae: AutoencoderKL = pipe.vae
    print('Before modifying',len(tokenizer))
    
    # Load ckpt from pretrained weights
    embedding_list, unet_list, concept_list = parse_new_concepts(concept_cfg)
    all_concept_emb, new_concept_cfg = merge_new_concepts_(embedding_list, concept_list, tokenizer, text_encoder)
    """
    "concept_list" is in the following format
    [
        {
            "lora_path": "experiments/cat2/models/edlora_model-latest.pth",
            "unet_alpha": 0.8,
            "text_encoder_alpha": 0.8,
            "concept_name": "<cat21> <cat22>"
        },
        {
            "lora_path": "experiments/wearable_glasses/models/edlora_model-latest.pth",
            "unet_alpha": 0.8,
            "text_encoder_alpha": 0.8,
            "concept_name": "<wearable> <glasses>"
        },
        {
            "lora_path": "experiments/watercolor/models/edlora_model-latest.pth",
            "unet_alpha": 1.0,
            "text_encoder_alpha": 1.0,
            "concept_name": "<watercolor1> <watercolor2>"
        }
    ]

    """
    """
    "new_concept_cfg" is in the following format
    {
    "<cat21>": 
        {
            "concept_token_ids": [49408, 49409, 49410, 49411, 49412, 49413, 49414, 49415, 49416, 49417, 49418, 49419, 49420, 49421, 49422, 49423], 
            "concept_token_names": ["<new0>", "<new1>", "<new2>", "<new3>", "<new4>", "<new5>", "<new6>", "<new7>", "<new8>", "<new9>", "<new10>", "<new11>", "<new12>", "<new13>", "<new14>", "<new15>"]
        }, 
    "<cat22>": 
        {
            "concept_token_ids": [49424, 49425, 49426, 49427, 49428, 49429, 49430, 49431, 49432, 49433, 49434, 49435, 49436, 49437, 49438, 49439], 
            "concept_token_names": ["<new16>", "<new17>", "<new18>", "<new19>", "<new20>", "<new21>", "<new22>", "<new23>", "<new24>", "<new25>", "<new26>", "<new27>", "<new28>", "<new29>", "<new30>", "<new31>"]},
    "<wearable>": 
        {"concept_token_ids": [49440, 49441, 49442, 49443, 49444, 49445, 49446, 49447, 49448, 49449, 49450, 49451, 49452, 49453, 49454, 49455], 
        "concept_token_names": ["<new32>", "<new33>", "<new34>", "<new35>", "<new36>", "<new37>", "<new38>", "<new39>", "<new40>", "<new41>", "<new42>", "<new43>", "<new44>", "<new45>", "<new46>", "<new47>"]}, 
    "<glasses>": 
        {
            "concept_token_ids": [49456, 49457, 49458, 49459, 49460, 49461, 49462, 49463, 49464, 49465, 49466, 49467, 49468, 49469, 49470, 49471], 
            "concept_token_names": ["<new48>", "<new49>", "<new50>", "<new51>", "<new52>", "<new53>", "<new54>", "<new55>", "<new56>", "<new57>", "<new58>", "<new59>", "<new60>", "<new61>", "<new62>", "<new63>"]}, 
    "<watercolor1>":
        {
            "concept_token_ids": [49472, 49473, 49474, 49475, 49476, 49477, 49478, 49479, 49480, 49481, 49482, 49483, 49484, 49485, 49486, 49487], 
            "concept_token_names": ["<new64>", "<new65>", "<new66>", "<new67>", "<new68>", "<new69>", "<new70>", "<new71>", "<new72>", "<new73>", "<new74>", "<new75>", "<new76>", "<new77>", "<new78>", "<new79>"]}, 
    "<watercolor2>": 
        {
            "concept_token_ids": [49488, 49489, 49490, 49491, 49492, 49493, 49494, 49495, 49496, 49497, 49498, 49499, 49500, 49501, 49502, 49503], 
            "concept_token_names": ["<new80>", "<new81>", "<new82>", "<new83>", "<new84>", "<new85>", "<new86>", "<new87>", "<new88>", "<new89>", "<new90>", "<new91>", "<new92>", "<new93>", "<new94>", "<new95>"]
        }
    }
    """
    all_concept_emb:Dict[str,torch.Tensor] # <conceptname>:Tensor
    new_concept_cfg:Dict[str,Dict] # <conceptname>:["conceptid":idlist(40967,40968...),"concept_token":token_list(new1, new2..)]
    print('After modifying',len(tokenizer))

    # NOTE: Get the embedding of prompt and compute weight of loras to be merged
    

    weight = compute_weight_to_combine_lora( tokenizer=tokenizer, text_encoder=text_encoder, 
                                            prompt = my_test_prompt, all_concept_cfg = new_concept_cfg,
                                            all_concept_emb = all_concept_emb, concept_list = concept_list)
    
    unet =  merge_unet(unet,weight,unet_list)

    
   
    # Run simple test
    pretrained_path = 'experiments/composed_edlora/chilloutmix/cat2_wearable_glasses_watercolor_chilloutmix/combined_model_base'
    pipe = EDLoRAPipeline( vae=vae,text_encoder=text_encoder,tokenizer = tokenizer,\
                            unet=unet, scheduler=test_scheduler).to('cuda')
    with open(f'{pretrained_path}/new_concept_cfg.json', 'r') as fr:
        new_concept_cfg = json.load(fr)
    pipe.set_new_concept_cfg(new_concept_cfg)
    
    #type(pipe) # EDLoRAPipeline
    negative_prompt = 'half body, not full body, cropped'
    image = pipe(my_test_prompt, negative_prompt=negative_prompt, height=768, width=768, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(21), guidance_scale=12).images[0]
    
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
    image.save(f'FUCKING_FINISHED.jpg')



# MOS
def mos():
    pretrained_model_path = 'experiments/composed_edlora/chilloutmix/cat2_wearable_glasses_watercolor_chilloutmix/combined_model_base'
    enable_edlora = True  # True for edlora, False for lora

    pipe = EDLoRAPipeline.from_pretrained(pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler'), torch_dtype=torch.float16).to('cuda')
    with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
        new_concept_cfg = json.load(fr)
    pipe.set_new_concept_cfg(new_concept_cfg)

    
    #TOK = '<wearable> <glasses>'  # the TOK is the concept name when training lora/edlora
    TOK = '<watercolor1> <watercolor2>'  # the TOK is the concept name when training lora/edlora
    #prompt = f'a {TOK} in front of the door'
    prompt = 'A <watercolor1> <watercolor2> painting of a <cat21> <cat22> in a <watercolor1> <watercolor2> style wearing <wearable> <glasses>' 
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
    image.save(f'_deleteit.jpg')
cifc()
#mos()
a = torch.load('/home/remote/tchuang/DLCV_final/Mix-of-Show/output/task_2/models/edlora_model-latest.pth')
for k in a['params']['unet'].keys():
    pass
    #print(k)
