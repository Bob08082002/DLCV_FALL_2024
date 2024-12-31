import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from PIL import Image
import clip
import torch
import language_evaluation

def readJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except:
        return None


def getGTCaptions(annotations):
    img_id_to_name = {}
    for img_info in annotations["images"]:
        img_name = img_info["file_name"].replace(".jpg", "")
        img_id_to_name[img_info["id"]] = img_name

    img_name_to_gts = defaultdict(list)
    for ann_info in annotations["annotations"]:
        img_id = ann_info["image_id"]
        img_name = img_id_to_name[img_id]
        img_name_to_gts[img_name].append(ann_info["caption"])
    return img_name_to_gts


class CIDERScore:
    def __init__(self):
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])

    def __call__(self, predictions, gts):
        predicts = []
        answers = []
        for img_name in predictions.keys():
            predicts.append(predictions[img_name])
            answers.append(gts[img_name])
        
        results = self.evaluator.run_evaluation(predicts, answers)
        return results['CIDEr']


class CLIPScore:
    def __init__(self):
        #self.device = "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def __call__(self, predictions, images_root):
        scores = {}

        for img_name, pred_caption in predictions.items():
            image_path = os.path.join(images_root, f"{img_name}.jpg")
            image = Image.open(image_path).convert("RGB")
            score = self.getCLIPScore(image, pred_caption[:])
            scores[img_name] = (pred_caption, score)
        
        # Sort scores in descending order
        sorted_scores = sorted(scores.items(), key=lambda x: x[1][1], reverse=True)
        
        # Print top-1 and last-1 results
        top1_img, (top1_caption, top1_score) = sorted_scores[0]
        last1_img, (last1_caption, last1_score) = sorted_scores[-1]
        
        print(f"\nTop-1 Image: {top1_img}")
        print(f"Caption: {top1_caption}")
        print(f"CLIPScore: {top1_score:.4f}\n")
        
        print(f"Last-1 Image: {last1_img}")
        print(f"Caption: {last1_caption}")
        print(f"CLIPScore: {last1_score:.4f}\n")
        
        # Return the average CLIP score
        total_score = sum(score for _, (_, score) in scores.items())
        return total_score / len(predictions)

    def getCLIPScore(self, image, caption):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([caption], truncate=True).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
        
        cos_sim = torch.nn.functional.cosine_similarity(image_features, text_features).item()
        return 2.5 * max(cos_sim, 0)


def main(args):
    # Read data
    predictions = readJSON(args.pred_file)
    annotations = readJSON(args.annotation_file)

    # Preprocess annotation file
    gts = getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CIDErScore
    cider_score = CIDERScore()(predictions, gts)

    # CLIPScore with top-1 and last-1 printing
    clip_score = CLIPScore()(predictions, args.images_root)
    
    print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--pred_file", help="Prediction json file")
    parser.add_argument("--images_root", default="p2_data/images/val/", help="Image root")
    parser.add_argument("--annotation_file", default="p2_data/val.json", help="Annotation json file")

    args = parser.parse_args()

    main(args)
