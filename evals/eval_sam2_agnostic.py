import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

## need to download this from sam2 repo
checkpoint = "sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
current_dir = os.getcwd()
# Move one level up to the parent directory
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# Change the working directory to the parent directory
os.chdir(parent_dir)
sys.path.append(parent_dir)
import random
import monai
import cv2
from os import makedirs
from os.path import join, basename
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
import json
from modules import DiscreteGroupImageCanonicalization, ESCNNEquivariantNetwork
from modules import TextPromptEncoder, MedSAMWithCanonicalization
from modules import sam_model_registry
from utils import FLanSDataset_pos_only, normalized_surface_distance_np, test_epoch_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")
# print("All warnings are suppressed!")
seed = 0
torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # To ensure deterministic behavior for certain operations
torch.backends.cudnn.benchmark = False  # Turn off optimization that may introduce randomness



with open("prompts/free_form_text_prompts.json", 'r') as file:
    label_dict = json.load(file)

with open("configs/test_data_paths.json", 'r') as file:
    test_data_path_lists = json.load(file)
    
with open("prompts/positional_free_form_text_prompts.json", 'r') as file:
    pos_prompt_dict = json.load(file)

with open("prompts/organ_positions.json", 'r') as file:
    pos_file_dict = json.load(file)
    
with open("prompts/organ_bounding_boxes.json", 'r') as file:
    pos_box_dict = json.load(file)
    
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def dice_coefficient(preds, targets, smooth=0.0):
    # Flatten the tensors
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    # Calculate Dice Coefficient
    intersection = (preds * targets).sum()
    dice_coeff = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    return dice_coeff.item()

def test_epoch_batch(eval_loader, predictor, epoch = None, prompt = "box"):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        dice_coeffs = []
        preds, trues = [], []

        # Only create a progress bar if this is the main process
        pbar = tqdm(eval_loader)
            
        for step, batch in enumerate(pbar):
            # one batch
            image, gt2D, tokens, text, box = batch["image"].to(device), batch["gt2D"].to(device), batch["token"].to(device),  batch["text"], batch['box']
            # forward pass
            image_np = image[0].permute(1,2,0).cpu().data.numpy()
            predictor.set_image(image_np)
            points = box[0].cpu().data.numpy()
            input_point = np.array([[(points[0]+points[2])/2, (points[1]+points[3])/2]])
            point_labels = np.array([1])
            if prompt == "point":
                # point prompt
                masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=point_labels, multimask_output=True)
            else:
                # box prompt
                masks, scores, _ = predictor.predict(point_coords=None, point_labels=None,box=points[None, :], multimask_output=False)

            pred = cv2.resize(masks[0], (256, 256), interpolation=cv2.INTER_LINEAR).reshape(1,1,256,256)
            preds.append(pred)
            trues.append(gt2D.cpu().data.numpy())
            # print(masks.shape, gt2D.shape)
            dice_coeffs.append(dice_coefficient(pred[0,0], gt2D.cpu().data.numpy()[0,0]))
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, eval total loss: {np.mean(dice_coeffs):.4f}")
        
    return np.mean(dice_coeffs), np.concatenate(preds, axis = 0), np.concatenate(trues, axis = 0)

with torch.no_grad(): 
    for key in test_data_path_lists.keys():
        
        test_dataset = FLanSDataset_pos_only(data_path_lists = {key: test_data_path_lists[key]},
                                      label_dict = label_dict, 
                                      pos_file_dict = pos_file_dict, 
                                      pos_prompt_dict = pos_prompt_dict,
                                      pos_box_dict = pos_box_dict)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


        print(key, "Box prompt")
        test_dice_score, test_preds, test_trues = test_epoch_batch(test_loader, predictor, prompt = "box")
        print(f"Dice: {test_dice_score:.3f}, NSD: {normalized_surface_distance_np(test_preds, test_trues, logits = False):.3f}")
        
        print(key, "Point prompt")
        test_dice_score, test_preds, test_trues = test_epoch_batch(test_loader, predictor, prompt = "point")
        print(f"Dice: {test_dice_score:.3f}, NSD: {normalized_surface_distance_np(test_preds, test_trues, logits = False):.3f}")
        print()
        
        