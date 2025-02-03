import os
import sys
import random
import json
import numpy as np
import torch
import yaml
import cv2
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils import FLanSDataset_pos_only, normalized_surface_distance_np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load SAM2 Model
checkpoint = "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load JSON files
def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

label_dict = load_json("prompts/free_form_text_prompts.json")
test_data_path_lists = load_json("configs/test_data_paths.json")
pos_prompt_dict = load_json("prompts/positional_free_form_text_prompts.json")
pos_file_dict = load_json("prompts/organ_positions.json")
pos_box_dict = load_json("prompts/organ_bounding_boxes.json")

# Load YAML Config
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Dice Coefficient Function
def dice_coefficient(preds, targets, smooth=0.0):
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

# Evaluation Function
def test_epoch_batch(eval_loader, predictor, epoch=None, prompt="box"):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        dice_coeffs, preds, trues = [], [], []
        pbar = tqdm(eval_loader)
        
        for step, batch in enumerate(pbar):
            image, gt2D, _, text, box = (
                batch["image"].to(device),
                batch["gt2D"].to(device),
                batch["token"].to(device),
                batch["text"],
                batch["box"],
            )
            
            # Convert image tensor to NumPy for prediction
            image_np = image[0].permute(1, 2, 0).cpu().numpy()
            predictor.set_image(image_np)
            points = box[0].cpu().numpy()
            
            if prompt == "point":
                input_point = np.array([[(points[0] + points[2]) / 2, (points[1] + points[3]) / 2]])
                point_labels = np.array([1])
                masks, scores, logits = predictor.predict(
                    point_coords=input_point, point_labels=point_labels, multimask_output=True
                )
            else:
                masks, scores, _ = predictor.predict(
                    point_coords=None, point_labels=None, box=points[None, :], multimask_output=False
                )

            # Resize prediction to match ground truth
            pred = cv2.resize(masks[0], (256, 256), interpolation=cv2.INTER_LINEAR).reshape(1, 1, 256, 256)
            preds.append(pred)
            trues.append(gt2D.cpu().numpy())

            # Compute Dice Score
            dice_coeffs.append(dice_coefficient(pred[0, 0], gt2D.cpu().numpy()[0, 0]))

            # Display progress
            pbar.set_description(
                f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Dice: {np.mean(dice_coeffs):.4f}"
            )

    return np.mean(dice_coeffs), np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

if __name__ == "__main__":
    with torch.no_grad():
        for key, data_paths in test_data_path_lists.items():
            test_dataset = FLanSDataset_pos_only(
                data_path_lists={key: data_paths},
                label_dict=label_dict,
                pos_file_dict=pos_file_dict,
                pos_prompt_dict=pos_prompt_dict,
                pos_box_dict=pos_box_dict
            )
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

            for prompt_type in ["box", "point"]:
                print(f"{key} - {prompt_type.capitalize()} Prompt")
                dice_score, preds, trues = test_epoch_batch(test_loader, predictor, prompt=prompt_type)
                nsd_score = normalized_surface_distance_np(preds, trues, logits=False)
                print(f"Dice: {dice_score:.3f}, NSD: {nsd_score:.3f}\n")
