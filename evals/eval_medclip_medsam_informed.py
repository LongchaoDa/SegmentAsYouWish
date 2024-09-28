import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TOKENIZERS_PARALLELISM"] = "False"
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
sys.path.append(parent_dir)
import random
import monai
from os import makedirs
from os.path import join, basename
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np
import torch
import yaml
import cv2
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
from torch.nn.parallel import DataParallel
import json
from modules import DiscreteGroupImageCanonicalization, ESCNNEquivariantNetwork
from modules import TextPromptEncoder, MedSAMWithCanonicalization
from modules import sam_model_registry
from utils import FLanSDataset, normalized_surface_distance_np, FLanSDataset_pos_only
import torch.nn.functional as F
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

### Load MedSAM
medsam_model = sam_model_registry["vit_b"](checkpoint="preload/sam_vit_b_01ec64.pth")
medsam_model = medsam_model.to(device)
medsam_model.eval()

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
    

def dice_coefficient(preds, targets, smooth=0.0):
    # Flatten the tensors
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    # Calculate Dice Coefficient
    intersection = (preds * targets).sum()
    dice_coeff = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    return dice_coeff.item()

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

### Load MedCLIP
from PIL import Image
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import MedCLIPProcessor
# load MedCLIP-ResNet50
MedCLIP = MedCLIPModel(vision_cls=MedCLIPVisionModel)
MedCLIP.from_pretrained()
MedCLIP.cuda()
print("MedCLIP Loaded")


def generate_random_bounding_boxes(image_shape, num_boxes):
    """
    Generate a set of random bounding boxes with various sizes and locations.

    Parameters:
    - image_shape: tuple (height, width)
    - num_boxes: int, number of bounding boxes to generate
    
    Returns:
    - List of bounding boxes in the format [(x_min, y_min, x_max, y_max), ...]
    """
    height, width = image_shape
    boxes = []
    
    for _ in range(num_boxes):
        # Randomly generate two points (x_min, y_min) and (x_max, y_max)
        x_min = random.randint(0, width - 300)
        y_min = random.randint(0, height - 300)

        # Randomly generate width and height of the box ensuring it fits within the image
        box_width = random.randint(100, 300)
        box_height = random.randint(100, 300)

        x_max = x_min + box_width
        y_max = y_min + box_height

        boxes.append((x_min, y_min, x_max, y_max))

    return boxes

def generate_grid_bounding_boxes(image_shape, box_size, num_boxes_per_row, num_boxes_per_col):
    """
    Generate a grid of bounding boxes from left to right, top to bottom.

    Parameters:
    - image_shape: tuple (height, width) of the image.
    - box_size: int, size of the bounding box (box_width = box_height = box_size).
    - num_boxes_per_row: int, number of boxes per row.
    - num_boxes_per_col: int, number of boxes per column.

    Returns:
    - List of bounding boxes in the format [(x_min, y_min, x_max, y_max), ...]
    """
    height, width = image_shape
    boxes = []

    # Loop through the grid and create the bounding boxes
    for row in range(num_boxes_per_col):
        for col in range(num_boxes_per_row):
            x_min = col * box_size
            y_min = row * box_size
            x_max = x_min + box_size
            y_max = y_min + box_size

            # Append bounding box coordinates to the list
            boxes.append((x_min, y_min, x_max, y_max))

    return boxes


def test_epoch_batch(eval_loader, epoch = None):
    processor = MedCLIPProcessor()
    with torch.no_grad(): 
        # get all possible segmentations
        dice_coeffs, preds, trues = [], [], []
        pbar = tqdm(eval_loader)
        
        boxes = generate_random_bounding_boxes((1024, 1024), 300) + generate_grid_bounding_boxes((1024, 1024), 128, 8, 8) + generate_grid_bounding_boxes((1024, 1024), 256, 4, 4) +  generate_grid_bounding_boxes((1024, 1024), 512, 2, 2)

        for step, batch in enumerate(pbar):
            img, gt2D, text, organ_name = batch["image"].to(device), batch["gt2D"].to(device),  batch["text"], batch["organ_name"]

            # generate all the masks for the random boxes
            medsam_segs = []
            image_embedding = medsam_model.image_encoder(img.to(device))
            for box in boxes:
                seg = medsam_inference(medsam_model, image_embedding, [box], 1024, 1024)
                medsam_segs.append(seg)
            medsam_segs = np.array(medsam_segs)

            # get cropped_img_embedding
            cropped_img_embeddings = []
            for i in range(medsam_segs.shape[0]):
                inputs = processor(
                    text=text[0], 
                    images= (torch.from_numpy(medsam_segs[i:i+1]).to(device).unsqueeze(1) + img)[0].cpu().numpy(), #
                    return_tensors="pt", 
                    padding=True
                    )
                outputs = MedCLIP(**inputs)
                cropped_img_embeddings.append(outputs['img_embeds'])
                
            cropped_img_embeddings = torch.cat(cropped_img_embeddings, dim = 0)
            text_embeddings = outputs['text_embeds']
            
            # print(text_embeddings.shape, cropped_img_embeddings.shape)
            cropped_img_embeddings /= cropped_img_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            # print(text_embeddings.shape, cropped_img_embeddings.shape)

            
            similarity = F.cosine_similarity(text_embeddings, cropped_img_embeddings, dim=-1) 
            best_idx = torch.argmax(similarity)
            final_mask = medsam_segs[best_idx]
            final_mask = cv2.resize(
                    final_mask,
                    (256, 256),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.uint8)

            preds.append(final_mask.reshape(gt2D.shape))
            trues.append(gt2D.cpu().data.numpy())
            dice_coeffs.append(dice_coefficient(final_mask.reshape(gt2D.shape), gt2D.cpu().data.numpy()))
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, eval total loss: {np.mean(dice_coeffs):.4f}")
            # break
        return np.mean(dice_coeffs), np.concatenate(preds, axis = 0), np.concatenate(trues, axis = 0), dice_coeffs
    
    
    
# Anatomy-informed
print("Anatomy-informed testing")
with torch.no_grad(): 
    for key in list(test_data_path_lists.keys()):
        test_dataset = FLanSDataset(data_path_lists = {key: test_data_path_lists[key]}, label_dict = label_dict, data_aug = False)
        test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=1, pin_memory=True)
        aug_test_dataset = FLanSDataset(data_path_lists = {key: test_data_path_lists[key]}, label_dict = label_dict, data_aug = True, group_order = 8)
        aug_test_loader = DataLoader(aug_test_dataset, batch_size = 1, shuffle=False, num_workers=1, pin_memory=True)
        test_dice_score, test_preds, test_trues, dice_coeffs = test_epoch_batch(test_loader)
        aug_test_dice_score, aug_test_preds, aug_test_trues, aug_dice_coeffs = test_epoch_batch(aug_test_loader)
        print(key)
        print(f"Dice: {test_dice_score:.3f},  NSD: {normalized_surface_distance_np(test_preds, test_trues):.3f}")
        print(f"Aug Dice: {aug_test_dice_score:.3f}, Aug NSD: {normalized_surface_distance_np(aug_test_preds, aug_test_trues):.3f}\n")
        print()
        torch.save({"test_set": [test_dice_score, test_preds, test_trues, dice_coeffs],
                    "aug_test_set": [aug_test_dice_score, aug_test_preds, aug_test_trues, aug_dice_coeffs]},
                    key +"medclip_medsam_informed.pt")