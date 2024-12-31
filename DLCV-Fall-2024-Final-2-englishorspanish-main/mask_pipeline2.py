import argparse
import hashlib
import json
import os.path
from datetime import datetime
import torch
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
import os
import cv2
import numpy as np
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, predict as dino_predict, annotate, load_image
from segment_anything import sam_model_registry, SamPredictor
import groundingdino.datasets.transforms as T

import random
import sys
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QInputDialog, QMessageBox

def postprocess_edges(image, edge_thickness=1):
    image_np = np.array(image)
    edges = cv2.Canny(image_np, 10, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=edge_thickness)
    postprocessed_image = Image.fromarray(edges)
    return postprocessed_image

def get_bbox(bbox_str):
    bbox_values = bbox_str.strip('[]').split(',')
    minY = float(bbox_values[0])
    minX = float(bbox_values[1])
    maxY = float(bbox_values[2])
    maxX = float(bbox_values[3])
    return [minY, minX, maxY, maxX]

def crop_object_from_bbox(mask, bbox):
    minY, minX, maxY, maxX = bbox
    object_crop = mask.crop((minX, minY, maxX, maxY))
    return object_crop

def project_to_target_bbox(cropped_object, target_bbox):
    minY, minX, maxY, maxX = target_bbox
    target_width = maxX - minX
    target_height = maxY - minY
    cropped_resized = cropped_object.resize((target_width, target_height))
    return cropped_resized, (minX, minY, maxX, maxY)

def process_objects(maskes, bounding_boxes, target_bboxes, H, W):
    final_canvas = np.zeros((H, W), dtype=np.uint8)
    for mask, bounding_box, target_bbox in zip(maskes, bounding_boxes, target_bboxes):
        mask = Image.fromarray(mask).convert('L')
        bbox = get_bbox(bounding_box)
        cropped_object = crop_object_from_bbox(mask, bbox)
        cropped_resized, target_position = project_to_target_bbox(cropped_object, target_bbox)
        cropped_array = np.array(cropped_resized)
        canvas_array = final_canvas[
            target_position[1]:target_position[3], 
            target_position[0]:target_position[2]
        ]
        combined_array = np.maximum(canvas_array, cropped_array)
        final_canvas[
            target_position[1]:target_position[3], 
            target_position[0]:target_position[2]
        ] = combined_array

    final_canvas = Image.fromarray(final_canvas)
    final_canvas = postprocess_edges(final_canvas, edge_thickness=2)
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


########################################
# PyQt Integration Code
########################################
import sys
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QImage, QColor
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QInputDialog, QMessageBox

class BoundingBoxSelector(QMainWindow):
    def __init__(self, num, W, H):
        super().__init__()
        self.setWindowTitle("Bounding Box Selector")
        self.setGeometry(100, 100, W, H)  # Set canvas size to WxH

        self.num = num  # Maximum number of bounding boxes
        self.current_count = 0  # Current number of bounding boxes

        self.start_point = None
        self.end_point = None
        self.bounding_box = None

        # Create a black image as background
        self.image = QImage(W, H, QImage.Format_RGB888)
        self.image.fill(QColor('black'))

        self.bbx_list = []

        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 492, 20)
        self.label.setStyleSheet("color: red;")
        self.label.setWordWrap(True)
        self.label.raise_()

    def map_to_image(self, point):
        # Since image and canvas size are the same, no scaling needed
        return point.x(), point.y()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.current_count >= self.num:
                self.show_max_reached_dialog()
                return
            self.start_point = event.pos()
            self.end_point = self.start_point

    def mouseMoveEvent(self, event):
        if self.start_point and event.buttons() == Qt.LeftButton:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_point:
            if self.current_count >= self.num:
                self.show_max_reached_dialog()
                return

            self.end_point = event.pos()
            mapped_start = self.map_to_image(self.start_point)
            mapped_end = self.map_to_image(self.end_point)

            ymin = min(mapped_start[1], mapped_end[1])
            xmin = min(mapped_start[0], mapped_end[0])
            ymax = max(mapped_start[1], mapped_end[1])
            xmax = max(mapped_start[0], mapped_end[0])

            self.bounding_box = (ymin, xmin, ymax, xmax)
            self.current_count += 1
            self.bbx_list.append(self.bounding_box)

            print(f"Bounding Box: (ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax})")
            self.label.setText(f"Bounding Box: (ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax})")

            self.start_point = None
            self.end_point = None
            self.update()

            # If we have reached the required number of boxes, show dialog
            if self.current_count == self.num:
                self.show_max_reached_dialog()

    def show_max_reached_dialog(self):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Maximum Bounding Boxes Reached")
        dialog.setText("You have reached the maximum number of bounding boxes.")
        dialog.setStandardButtons(QMessageBox.Retry | QMessageBox.Ok)
        dialog.setDefaultButton(QMessageBox.Ok)

        choice = dialog.exec_()

        if choice == QMessageBox.Retry:
            self.current_count = 0
            self.bbx_list = []
            self.label.setText("Bounding boxes reset. You can start over.")
            self.update()
        elif choice == QMessageBox.Ok:
            print("Final Bounding Boxes:")
            for box in self.bbx_list:
                print(box)
            self.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.image.isNull():
            painter.drawImage(0, 0, self.image)

        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        for box in self.bbx_list:
            ymin, xmin, ymax, xmax = box
            painter.drawRect(xmin, ymin, xmax - xmin, ymax - ymin)

        if self.start_point and self.end_point:
            temp_start = self.map_to_image(self.start_point)
            temp_end = self.map_to_image(self.end_point)
            temp_ymin = min(temp_start[1], temp_end[1])
            temp_xmin = min(temp_start[0], temp_end[0])
            temp_ymax = max(temp_start[1], temp_end[1])
            temp_xmax = max(temp_start[0], temp_end[0])
            painter.drawRect(temp_xmin, temp_ymin, temp_xmax - temp_xmin, temp_ymax - temp_ymin)

def get_user_defined_bboxes(num_bboxes, W, H):
    # Run a PyQt Application to get bounding boxes
    app = QApplication(sys.argv)
    window = BoundingBoxSelector(num_bboxes, W, H)
    window.show()
    app.exec_()  # block until user closes

    # window.bbx_list contains [(ymin,xmin,ymax,xmax), ...]
    return window.bbx_list


########################################


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    pretrained_model_path = args.pretrained_model
    enable_edlora = True

    H = args.image_height
    W = args.image_width

    concept_image = []
    concept_mask = []
    concept_bboxes = []

    regions = args.prompt_rewrite.split('|') 
    replace_token = args.token.split('|') 
    print(replace_token)
    num_concept = len(replace_token)
    idx = 0

    while(idx < num_concept):
        TOK = replace_token[idx].replace('[', '').replace(']', '')
        print(f"process token {TOK}")
        region = regions[idx]
        if region == '':
            break
        prompt_region, neg_prompt_region, pos = region.split('-*-')
        prompt_region = prompt_region.replace('[', '').replace(']', '')
        neg_prompt_region = neg_prompt_region.replace('[', '').replace(']', '')

        from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline
        pipe = EDLoRAPipeline.from_pretrained(pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler'), torch_dtype=torch.float16).to('cuda')
        with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
            new_concept_cfg = json.load(fr)
        pipe.set_new_concept_cfg(new_concept_cfg)

        prompt = prompt_region
        negative_prompt = neg_prompt_region
        image = pipe(prompt_region, negative_prompt=neg_prompt_region, height=512, width=512, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(args.seed), guidance_scale=7.5).images[0]
        #image.save(f"{idx}.png")
        image = np.asarray(image)
        concept_image.append(image)

        print("Single Box generation ...")
        concept_name = TOK
        groundingdino_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        groundingdino_checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
        sam_checkpoint = "./sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        box_threshold = 0.25
        text_threshold = 0.25

        dino_model = load_model(groundingdino_config_path, groundingdino_checkpoint_path, device=device)
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)

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

        boxes = boxes * torch.Tensor([W, H, W, H])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        predictor.set_image(image_rgb)

        # minX = xyxy[0][0]
        # minY = xyxy[0][1]
        # maxX = xyxy[0][2]
        # maxY = xyxy[0][3]

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
            box=input_box,  
            multimask_output=False
        )

        mask = masks[0]
        mask_uint8 = (mask.astype(np.uint8) * 255)
        #cv2.imwrite(f"{idx}_mask.png", mask_uint8)
        edges = cv2.Canny(mask_uint8, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges_thicker = cv2.dilate(edges, kernel, iterations=1)

        concept_mask.append(edges_thicker)
        concept_bboxes.append(bounding_box)

        idx += 1
        print(f"{concept_name} mask generation finish")

    ###############################TODO : LLM BOX GEN #################################
    # MASK GUI
    target_bboxes_user = get_user_defined_bboxes(num_concept, args.image_width, args.image_height)
    target_bboxes = target_bboxes_user

    ################################ MASK　PROJECTION #################################
    print("Box to Box Projection ...")
    merged_mask = process_objects(concept_mask, concept_bboxes, target_bboxes, args.image_height, args.image_width)

    output_dir = args.mask_dir
    os.makedirs(output_dir, exist_ok=True)
    filename = '1'
    merged_mask.save(args.sketch_condition)
    bbox_info_file = os.path.join(output_dir, f"{filename}.txt")

    with open(bbox_info_file, 'w') as f:
        for idx, token in enumerate(replace_token):
            minY, minX, maxY, maxX = target_bboxes[idx]
            f.write(f"{token}: [{minY}, {minX}, {maxY}, {maxX}]\n")

    print("Finish !!")
