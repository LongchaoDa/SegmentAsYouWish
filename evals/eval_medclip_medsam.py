import os
import sys
import random
import torch
import numpy as np
import json
import cv2
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from modules import sam_model_registry
from utils import FLanSDataset, normalized_surface_distance_np
from medclip import MedCLIPModel, MedCLIPVisionModel, MedCLIPProcessor
import warnings

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
seed = 0
torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior for certain operations
torch.backends.cudnn.benchmark = False  # Turns off optimization that may introduce randomness

# Load MedSAM
medsam_model = sam_model_registry["vit_b"](checkpoint="preload/sam_vit_b_01ec64.pth").to(device).eval()

# Load MedCLIP
MedCLIP = MedCLIPModel(vision_cls=MedCLIPVisionModel)
MedCLIP.from_pretrained()
MedCLIP.cuda()
MedCLIP.eval()
print("MedCLIP Loaded")

# Load JSON configuration files
with open("prompts/free_form_text_prompts.json", 'r') as file:
    label_dict = json.load(file)

with open("configs/test_data_paths.json", 'r') as file:
    test_data_path_lists = json.load(file)


def dice_coefficient(preds, targets, smooth=0.0):
    """Compute Dice coefficient for binary segmentation masks."""
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    intersection = (preds * targets).sum()
    dice_coeff = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice_coeff.item()


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    """Generate segmentation using MedSAM model."""
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None
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
    return (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


def test_epoch_batch(eval_loader, epoch=None):
    """Evaluate model performance over an epoch."""
    processor = MedCLIPProcessor()
    dice_coeffs, preds, trues = [], [], []
    pbar = tqdm(eval_loader)

    for step, batch in enumerate(pbar):
        img, gt2D, text, _ = batch["image"].to(device), batch["gt2D"].to(device), batch["text"], batch["organ_name"]

        # Generate segmentation mask using MedSAM
        image_embedding = medsam_model.image_encoder(img.to(device))
        medsam_seg = medsam_inference(medsam_model, image_embedding, [(0, 0, 1024, 1024)], 1024, 1024)

        # Get MedCLIP embeddings
        inputs = processor(text=text[0], images=img.cpu().numpy(), return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = MedCLIP(**inputs)

        cropped_img_embeddings = outputs['img_embeds']
        text_embeddings = outputs['text_embeds']

        # Compute similarity and select best mask
        similarity = F.cosine_similarity(text_embeddings, cropped_img_embeddings, dim=-1)
        best_idx = torch.argmax(similarity)
        final_mask = medsam_seg[best_idx]

        final_mask = cv2.resize(final_mask, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        preds.append(final_mask.reshape(gt2D.shape))
        trues.append(gt2D.cpu().numpy())
        dice_coeffs.append(dice_coefficient(final_mask.reshape(gt2D.shape), gt2D.cpu().numpy()))

        pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Eval Dice: {np.mean(dice_coeffs):.4f}")

    return np.mean(dice_coeffs), np.concatenate(preds, axis=0), np.concatenate(trues, axis=0), dice_coeffs


if __name__ == "__main__":
    print("Anatomy-informed testing")

    with torch.no_grad():
        for key, data_paths in test_data_path_lists.items():
            for data_aug in [False, True]:  # Loop over original and augmented datasets
                dataset = FLanSDataset(
                    data_path_lists={key: data_paths},
                    label_dict=label_dict,
                    data_aug=data_aug,
                    group_order=8 if data_aug else None
                )
                test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

                dice_score, preds, trues, dice_coeffs = test_epoch_batch(test_loader)

                aug_status = "Augmented" if data_aug else "Original"
                print(f"{key} - {aug_status} Dice: {dice_score:.3f}, NSD: {normalized_surface_distance_np(preds, trues):.3f}")
