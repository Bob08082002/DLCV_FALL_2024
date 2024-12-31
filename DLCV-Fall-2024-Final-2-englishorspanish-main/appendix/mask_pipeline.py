import argparse
import hashlib
import json
import os.path
from datetime import datetime
import torch
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline
import os
import cv2
import numpy as np
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, predict as dino_predict, annotate, load_image
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T

import random

def postprocess_edges(image, edge_thickness=1):
    # Convert the image to a NumPy array for processing
    image_np = np.array(image)
    
    # Apply Canny edge detection to find edges
    edges = cv2.Canny(image_np, 30, 200)

    # Convert edges to a binary mask (0 or 255)
    kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for dilation
    edges = cv2.dilate(edges, kernel, iterations=edge_thickness)  # Dilate the edges to increase thickness
    
    # Convert back to PIL image
    postprocessed_image = Image.fromarray(edges)
    return postprocessed_image

def get_bbox(bbox_str):
        
    # Remove the square brackets and split the values by comma
    bbox_values = bbox_str.strip('[]').split(',')
    
    # Convert values to float or int
    minY = float(bbox_values[0])  # You can change to int if needed
    minX = float(bbox_values[1])
    maxY = float(bbox_values[2])
    maxX = float(bbox_values[3])
    
    return [minY, minX, maxY, maxX]

def crop_object_from_bbox(mask, bbox):
    # Crop the object from the mask using the bounding box
    minY, minX, maxY, maxX = bbox
    object_crop = mask.crop((minX, minY, maxX, maxY))
    return object_crop

def project_to_target_bbox(cropped_object, target_bbox):
    # Get the target position (minY, minX, maxY, maxX)
    minY, minX, maxY, maxX = target_bbox
    target_width = maxX - minX
    target_height = maxY - minY
    
    # Resize the cropped object to fit the target bounding box
    cropped_resized = cropped_object.resize((target_width, target_height))
    
    return cropped_resized, (minX, minY, maxX, maxY)

def process_objects(maskes, bounding_boxes, target_bboxes, H, W):
    # Initialize a blank canvas (512x512) as a NumPy array
    final_canvas = np.zeros((H, W), dtype=np.uint8)
    
    # Loop through each directory and target bounding box
    for mask, bounding_box, target_bbox in zip(maskes, bounding_boxes, target_bboxes):
        
        mask = Image.fromarray(mask).convert('L')
        # Get the mask and bounding box for the chosen object
        bbox = get_bbox(bounding_box)
        
        # Crop the object based on the bounding box
        cropped_object = crop_object_from_bbox(mask, bbox)
        
        # Project the object to the target bounding box and resize it
        cropped_resized, target_position = project_to_target_bbox(cropped_object, target_bbox)
        
        # Convert the resized object and canvas to NumPy arrays
        cropped_array = np.array(cropped_resized)
        canvas_array = final_canvas[
            target_position[1]:target_position[3], 
            target_position[0]:target_position[2]
        ]
        
        # Only keep the white parts (non-zero) from the new object
        combined_array = np.maximum(canvas_array, cropped_array)
        
        # Update the corresponding part of the canvas
        final_canvas[
            target_position[1]:target_position[3], 
            target_position[0]:target_position[2]
        ] = combined_array
    
    # Convert the final canvas back to a PIL image
    final_canvas = Image.fromarray(final_canvas)
    
    # Postprocess edges to ensure uniform thickness
    final_canvas = postprocess_edges(final_canvas, edge_thickness=1)
    
    return final_canvas

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
    parser.add_argument('--mask_dir', default='final_dataset/merge_mask/trash', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    ############################## LOAD CFG and MODEL ################################
    args = parse_args()
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    pretrained_model_path = args.pretrained_model
    enable_edlora = True  # True for edlora, False for lora

    H = args.image_height
    W = args.image_width
    ###################################################################################

    #################### GENERATE SINGLE CONCEPT IMAGE AND MASK #######################
    concept_image = []
    concept_mask = []
    concept_bboxes = []

    regions = args.prompt_rewrite.split('|') # split the different region
    replace_token = args.token.split('|') 

    print(replace_token)
    num_concept = len(replace_token)
    idx = 0

    while(idx < num_concept):

        TOK = replace_token[idx].replace('[', '').replace(']', '')  # the TOK is the concept name when training lora/edlora

        print(f"process token {TOK}")

        region = regions[idx]

        if region == '':
            break
        prompt_region, neg_prompt_region, pos = region.split('-*-')
        prompt_region = prompt_region.replace('[', '').replace(']', '')
        neg_prompt_region = neg_prompt_region.replace('[', '').replace(']', '')
        # sample
        pipe = EDLoRAPipeline.from_pretrained(pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler'), torch_dtype=torch.float16).to('cuda')
        with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
            new_concept_cfg = json.load(fr)

        pipe.set_new_concept_cfg(new_concept_cfg)

        
        prompt = prompt_region
        negative_prompt = neg_prompt_region

        image = pipe(prompt_region, negative_prompt=neg_prompt_region, height=H, width=W, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(1), guidance_scale=7.5).images[0]
        image = np.asarray(image)
        concept_image.append(image)

        ########## SINGLE BOX GEN ##########
        print("Single Box generation ...")
        concept_name = TOK
        # GroundingDINO model and checkpoint
        groundingdino_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        groundingdino_checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

        # SAM model
        sam_checkpoint = "./sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        # Thresholds
        box_threshold = 0.25
        text_threshold = 0.25

        ########################################
        # Load GroundingDINO Model
        ########################################
        dino_model = load_model(groundingdino_config_path, groundingdino_checkpoint_path, device=device)

        ########################################
        # Load SAM
        ########################################
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)

        ########################################
        # Processing All Images in Directory
        ########################################
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        caption = concept_name

        
        H, W, _ = image.shape
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        image_source = np.asarray(image_rgb)
        image_source = Image.fromarray(image_source)
        image, _ = transform(image_source, None)
        
        # GroundingDINO Inference
        boxes, logits, phrases = dino_predict(dino_model, image, caption=caption, box_threshold=box_threshold, text_threshold=text_threshold, device=device)

        if boxes is None or len(boxes) == 0:
            print(f"No objects found in concept {TOK}")
            continue

        # Annotated image visualization
        # annotated_image= annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        boxes = boxes * torch.Tensor([W, H, W, H])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Prepare SAM input
        predictor.set_image(image_rgb)

        minX = W
        minY = H
        maxX = 0
        maxY = 0
        for i in xyxy:
            if minX > i[0]:
                minX = i[0]
            if minY > i[1]:
                minY = i[1]
            if maxX < i[2]:
                maxX = i[2]
            if maxY < i[3]:
                maxY = i[3]

        input_box = np.array([minX, minY, maxX, maxY], dtype=np.float32)
        bounding_box = f"[{minY}, {minX}, {maxY}, {maxX}]"

        masks, scores, logits = predictor.predict(
            box=input_box,  # Add batch dimension
            multimask_output=False
        )

        mask = masks[0]  # Use the first mask
        mask_uint8 = (mask.astype(np.uint8) * 255)
        
        # Perform edge detection to create sketch
        edges = cv2.Canny(mask_uint8, 100, 150)

        # Dilate the edges to make them thicker
        kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel for dilation
        edges_thicker = cv2.dilate(edges, kernel, iterations=2)
        
        
        concept_mask.append(edges_thicker)
        concept_bboxes.append(bounding_box)

        idx += 1
        print(f"{concept_name} mask generation finish")
    ###################################################################################

    ###############################TODO : LLM BOX GEN #################################

    # INPUT: prompt, token num, Image width & height
    # prompt -> args.prompt
    # different concept image -> concept_image[idx]
    # Image infomation -> H, W

    # OUTPUT: Target Bounding BOX for each concept 
    # BOX -> target_bboxes[idx]

    #target_bboxes = []
    ############## test bounding box ############
    target_bboxes = [
        [80, 30, 190, 130],  # Bounding box for object 1
        [80, 190, 190, 290],  # Bounding box for object 2
        [80, 350, 190, 450],  # Bounding box for object 3
    ]

    ###################################################################################

    ################################ MASK　PROJECTION #################################
    print("Box to Box Projection ...")
    merged_mask = process_objects(concept_mask, concept_bboxes, target_bboxes, H, W)

    ###################################################################################


    ################################ IMAGE　GENERATION ################################
    # due to CUDA memory issue, split the file into two parts
    
    output_dir = args.mask_dir
    os.makedirs(output_dir, exist_ok=True)
    # here output all the file to do the next part
    filename = '0'
    output_filename = filename + ".png"
    merged_mask.save(args.sketch_condition)
    # write bounding box infomation
    bbox_info_file = os.path.join(output_dir, f"{filename}.txt")

    with open(bbox_info_file, 'w') as f:
        for idx, token in enumerate(replace_token):
            minY, minX, maxY, maxX = target_bboxes[idx]
            f.write(f"{token}: [{minY}, {minX}, {maxY}, {maxX}]\n")

    print("Finish !!")
    ###################################################################################

    
        

        
