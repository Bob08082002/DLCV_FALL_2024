import os
import json
from PIL import Image
from tqdm import tqdm
import sys

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration



# load model and images and generate captions
def generate_captions_from_folder(image_folder, output_file):
    captions = {}
    
    for filename in tqdm(os.listdir(image_folder), desc="Processing images"):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Load the image
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)

            # --------------------------------- Define Language Instruction --------------------------------- 
            # Prepare the prompt for the model
            #conversation = [ # use apply_chat_template need jinja package
            #    {
            #        "role": "user",
            #        "content": [
            #            {"type": "text", "text": "A short image caption."},
            #            {"type": "image"},
            #        ],
            #    },
            #]
            prompt = "USER: <image>\nA short image caption ASSISTANT:"

            # Process image and prompt
            inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)

            # --------------------------------- Define generation config --------------------------------- 
            # Generate caption: beam search
            output = model.generate(
                **inputs, 
                max_new_tokens=25, 
                num_beams=3, 
                do_sample=False
            )
            """output = model.generate(
                **inputs, 
                max_new_tokens=20,      # Limit tokens for faster output
                do_sample=True,         # Enable sampling instead of deterministic search
                top_k=10,               # Limit token choices to the top 10, for faster sampling
                top_p=0.9,              # Nucleus sampling (optional, helps maintain quality while keeping randomness)
                temperature=0.7         # Slight temperature to add a bit of variability (optional, can be 1.0 for faster output)
            )"""
            caption = processor.decode(output[0][2:], skip_special_tokens=True)
            # Extract text after "ASSISTANT:" and save the generated caption
            caption = caption.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in caption else caption

            # Add caption to dictionary (use filename without extension as key)
            file_key = os.path.splitext(filename)[0]
            captions[file_key] = caption
            print(f"file name: {file_key} \ncaption: {caption}")

            

    # Write captions to JSON file
    with open(output_file, 'w') as f:
        json.dump(captions, f, indent=4)


if __name__ == '__main__':
    # ------------- Parameters ------------- 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    # Define the folder containing the images and the output file
    # get $1 $2
    image_folder = sys.argv[1]  # path to the folder containing test images
    output_file = sys.argv[2]  # path to the output json file

    # Load model and processor
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Run the caption generation
    generate_captions_from_folder(image_folder, output_file)