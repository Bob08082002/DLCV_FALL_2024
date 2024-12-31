# Report P3-1: conduct the CLIP-based zero shot classification on hw2_data/clip_zeroshot/val
# reference: https://github.com/openai/CLIP
#
import os
import sys
import json
import clip
import torch
from PIL import Image

# get $1, $2
image_folder = sys.argv[1]   # val image folder
jsonfile_path = sys.argv[2]   # path of id2label.json

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# Load id2label.json
with open(jsonfile_path, "r") as f:
    id2label = json.load(f)

# Prepare text inputs for all class labels. Use the prompt "A photo of {object}."
text_inputs = torch.cat([clip.tokenize(f"A photo of {label}.") for label in id2label.values()]).to(device)

# Counters and storage for successful/failed cases
correct = 0
total = 0
successful_cases = []
failed_cases = []

def classify_image(image_path):
    """Classify a single image using CLIP and return the result."""
    global correct, total

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Extract true label from the filename (e.g., "12_abc.png" -> 12 -> "skyscraper")
    true_class_id = int(os.path.basename(image_path).split("_")[0])
    true_label = id2label[str(true_class_id)]

    # Calculate image and text features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Calculate similarity and get top 5 predictions
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # Get the top-1 predicted label
    predicted_label = list(id2label.values())[indices[0]]

    # Update accuracy counters
    total += 1
    if predicted_label == true_label:
        correct += 1
        successful_cases.append((image_path, true_label, predicted_label))
    else:
        failed_cases.append((image_path, true_label, predicted_label))

    

    


# Classify all images in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        image_path = os.path.join(image_folder, filename)
        classify_image(image_path)

# Calculate and print accuracy
accuracy = correct / total * 100
print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")

# Print 5 successful cases
print("\n5 Successful Cases:")
for i, (path, true, pred) in enumerate(successful_cases[:5]):
    print(f"{i+1}. {os.path.basename(path)} - True: {true}, Predicted: {pred}")

# Print 5 failed cases
print("\n5 Failed Cases:")
for i, (path, true, pred) in enumerate(failed_cases[:5]):
    print(f"{i+1}. {os.path.basename(path)} - True: {true}, Predicted: {pred}")