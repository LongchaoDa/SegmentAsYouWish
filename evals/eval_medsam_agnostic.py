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
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings

# Custom Modules
from modules import sam_model_registry
from utils import FLanSDataset_pos_only, normalized_surface_distance_np

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
seed = 0
torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load MedSAM Model
medsam_model = sam_model_registry["vit_b"](checkpoint="preload/sam_vit_b_01ec64.pth").to(device).eval()

# Load JSON Files
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
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
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
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    return (low_res_pred > 0.5).astype(np.uint8)


def test_epoch_batch(eval_loader, model, epoch=None):
    with torch.no_grad():
        dice_coeffs, preds, trues = [], [], []
        pbar = tqdm(eval_loader)

        for step, batch in enumerate(pbar):
            img, gt2D, box = batch["image"].to(device), batch["gt2D"].to(device), batch["box"].to(device)
            image_embedding = medsam_model.image_encoder(img)
            medsam_pred = medsam_inference(medsam_model, image_embedding, box, H=256, W=256)

            preds.append(medsam_pred)
            trues.append(gt2D.cpu().numpy())
            dice_coeffs.append(dice_coefficient(medsam_pred, gt2D.cpu().numpy()))

            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Dice: {np.mean(dice_coeffs):.4f}")

    return np.mean(dice_coeffs), np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


if __name__ == "__main__":
    with torch.no_grad():
        for key in test_data_path_lists.keys():
            test_dataset = FLanSDataset_pos_only(
                data_path_lists={key: test_data_path_lists[key]},
                label_dict=label_dict,
                pos_file_dict=pos_file_dict,
                pos_prompt_dict=pos_prompt_dict,
                pos_box_dict=pos_box_dict
            )
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

            test_dice_score, test_preds, test_trues = test_epoch_batch(test_loader, medsam_model)
            print(key)
            print(f"Dice: {test_dice_score:.3f}, NSD: {normalized_surface_distance_np(np.expand_dims(test_preds, axis=1), test_trues):.3f}\n")
