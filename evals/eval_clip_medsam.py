import os
import random
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from modules import sam_model_registry
from utils import FLanSDataset, normalized_surface_distance_np

# Suppress warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure reproducibility
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

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load MedSAM model
medsam_model = sam_model_registry["vit_b"](checkpoint="preload/sam_vit_b_01ec64.pth").to(device).eval()

# Load JSON files
def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

label_dict = load_json("prompts/free_form_text_prompts.json")
test_data_path_lists = load_json("configs/test_data_paths.json")
pos_prompt_dict = load_json("prompts/positional_free_form_text_prompts.json")
pos_file_dict = load_json("prompts/organ_positions.json")
pos_box_dict = load_json("prompts/organ_bounding_boxes.json")


def dice_coefficient(preds, targets, smooth=0.0):
    preds, targets = preds.reshape(-1), targets.reshape(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device).unsqueeze(1)
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(points=None, boxes=box_torch, masks=None)
    
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(low_res_pred, size=(H, W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()
    return (low_res_pred > 0.5).astype(np.uint8)


def generate_random_bounding_boxes(image_shape, num_boxes):
    height, width = image_shape
    return [
        (
            random.randint(0, width - 300),
            random.randint(0, height - 300),
            random.randint(100, 300) + random.randint(0, width - 300),
            random.randint(100, 300) + random.randint(0, height - 300)
        )
        for _ in range(num_boxes)
    ]


def generate_grid_bounding_boxes(image_shape, box_size, num_boxes_per_row, num_boxes_per_col):
    return [
        (col * box_size, row * box_size, (col + 1) * box_size, (row + 1) * box_size)
        for row in range(num_boxes_per_col) for col in range(num_boxes_per_row)
    ]


def test_epoch_batch(eval_loader, epoch=None):
    boxes = (
        generate_random_bounding_boxes((1024, 1024), 300) + 
        generate_grid_bounding_boxes((1024, 1024), 128, 8, 8) + 
        generate_grid_bounding_boxes((1024, 1024), 256, 4, 4) +  
        generate_grid_bounding_boxes((1024, 1024), 512, 2, 2)
    )

    dice_coeffs, preds, trues = [], [], []
    pbar = tqdm(eval_loader, desc=f"Epoch {epoch}" if epoch else "Testing")

    with torch.no_grad():
        for batch in pbar:
            img, gt2D, text = batch["image"].to(device), batch["gt2D"].to(device), batch["text"]

            # Generate masks for random bounding boxes
            image_embedding = medsam_model.image_encoder(img)
            medsam_segs = np.array([medsam_inference(medsam_model, image_embedding, [box], 1024, 1024) for box in boxes])

            # Get cropped image embeddings
            cropped_img_embeddings = torch.cat([
                medsam_model.image_encoder(torch.from_numpy(seg[np.newaxis]).to(device).unsqueeze(1) * img).mean((-1, -2))
                for seg in medsam_segs
            ], dim=0)

            # Generate text embeddings
            inputs = clip_processor(text=text[0], return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            text_embeddings = clip_model.get_text_features(**inputs).reshape(1, 256, 2).mean(-1)

            # Compute similarity and select best mask
            similarity = F.cosine_similarity(text_embeddings, cropped_img_embeddings, dim=-1)
            final_mask = cv2.resize(medsam_segs[torch.argmax(similarity)], (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            preds.append(final_mask.reshape(gt2D.shape))
            trues.append(gt2D.cpu().numpy())
            dice_coeffs.append(dice_coefficient(final_mask.reshape(gt2D.shape), gt2D.cpu().numpy()))
            pbar.set_description(f"Epoch {epoch if epoch else 'Test'}, Dice: {np.mean(dice_coeffs):.4f}")

    return np.mean(dice_coeffs), np.concatenate(preds, axis=0), np.concatenate(trues, axis=0), dice_coeffs

if __name__ == "__main__":
    print("Anatomy-informed testing")

    with torch.no_grad():
        for key, data_paths in test_data_path_lists.items():
            for data_aug in [False, True]:  # Loop over both augmentation settings
                dataset = FLanSDataset(
                    data_path_lists={key: data_paths}, 
                    label_dict=label_dict, 
                    data_aug=data_aug, 
                    group_order=8 if data_aug else None  # Apply group_order only for augmented data
                )
                data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

                dice_score, preds, trues, dice_coeffs = test_epoch_batch(data_loader)

                aug_status = "Augmented" if data_aug else "Original"
                print(f"{key} - {aug_status} Dice: {dice_score:.3f}, NSD: {normalized_surface_distance_np(preds, trues):.3f}")
