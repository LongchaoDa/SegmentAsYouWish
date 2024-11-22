import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
from utils import FLanSDataset_pos_only, normalized_surface_distance_np, test_epoch_batch
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

def test_epoch_batch(eval_loader, model, epoch = None):
    with torch.no_grad():
        eval_losses, seg_losses, ce_losses, clf_losses, dice_coeffs = [], [], [], [], []
        preds, trues = [], []

        # Only create a progress bar if this is the main process
        pbar = tqdm(eval_loader)
            
        for step, batch in enumerate(pbar):
            # one batch
            img, gt2D, box = batch["image"].to(device), batch["gt2D"].to(device), batch["box"].to(device)
            # forward pass
            image_embedding = medsam_model.image_encoder(img.to(device))
            medsam_pred = medsam_inference(medsam_model, image_embedding, box.to(device), H = 256, W = 256)
            
            preds.append(medsam_pred)
            trues.append(gt2D.cpu().data.numpy())

            dice_coeffs.append(dice_coefficient(medsam_pred, gt2D.cpu().data.numpy()))
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, eval total loss: {np.mean(dice_coeffs):.4f}")
        
    return np.mean(dice_coeffs), np.concatenate(preds, axis = 0), np.concatenate(trues, axis = 0)

with torch.no_grad(): 
    for key in test_data_path_lists.keys():
        
        test_dataset = FLanSDataset_pos_only(data_path_lists = {key: test_data_path_lists[key]},
                                      label_dict = label_dict, 
                                      pos_file_dict = pos_file_dict, 
                                      pos_prompt_dict = pos_prompt_dict,
                                      pos_box_dict = pos_box_dict)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)


        test_dice_score, test_preds, test_trues = test_epoch_batch(test_loader, medsam_model)
        print(key)
        print(f"Dice: {test_dice_score:.3f}, NSD: {normalized_surface_distance_np(np.expand_dims(test_preds, axis = 1), test_trues):.3f}")
        print()