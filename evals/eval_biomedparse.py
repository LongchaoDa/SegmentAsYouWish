# Evaluate biomedparse on our test sets
# Need to install biomedparse first
from PIL import Image
import torch
from modeling.BaseModel import BaseModel
from modeling import build_model
from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES
from inference_utils.inference import interactive_infer_image
from inference_utils.output_processing import check_mask_stats
import numpy as np
import json
from utils import FLanSDataset, normalized_surface_distance_np, dice_coefficient
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load JSON configuration files
    with open("prompts/free_form_text_prompts.json", 'r') as file:
        label_dict = json.load(file)

    with open("configs/test_data_paths.json", 'r') as file:
        test_data_path_lists = json.load(file)

    with open("prompts/positional_free_form_text_prompts.json", 'r') as file:
        pos_prompt_dict = json.load(file)

    with open("prompts/organ_positions.json", 'r') as file:
        pos_file_dict = json.load(file)

    # Build model config
    opt = load_opt_from_config_files(["BiomedParse/configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)

    # Load pretrained model
    pretrained_pth = 'hf_hub:microsoft/BiomedParse'
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    # Precompute text embeddings for evaluation
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

        # Iterate over data transformations and test datasets
        for prompt_type in ["organ_name", "free-form"]:
            print("="*20)
            print(f"Prompt Type: {prompt_type}")
            for data_transform in ["original", "transform"]:
                for key, data_path in test_data_path_lists.items():
                    # Select dataset type based on transformation
                    dataset = FLanSDataset(
                        data_path_lists={key: data_path}, 
                        label_dict=label_dict, 
                        data_aug=(data_transform == "transform"), 
                        group_order=4 if data_transform == "transform" else None
                    )
                    test_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

                    i = 0
                    dices, nsds = [], []
                    for batch in test_loader:
                        i += 1
                        text_prompt = batch['text'][0] if prompt_type == "free-form" else batch['organ_name'][0]
                        image = batch["image"].cuda().permute(0, 2, 3, 1)[0]
                        gt_mask = batch["gt2D"][0, 0].cuda()

                        # Normalize image and convert to PIL format
                        image_array = (image.cpu().numpy() - image.min().cpu().numpy()) / (image.max().cpu().numpy() - image.min().cpu().numpy())
                        image_array = torch.from_numpy((image_array * 255).astype(np.uint8)).to(device)

                        # Generate predicted mask
                        pred_mask = interactive_infer_image(model, image_array, text_prompt)[0, ::4, ::4]

                        # Compute Dice coefficient and NSD
                        dice_score = dice_coefficient(torch.from_numpy(pred_mask), gt_mask.cpu(), apply_sigmoid=False)
                        nsd_score = normalized_surface_distance_np(pred_mask, gt_mask.cpu().numpy(), threshold=0.5, distance_threshold=1.5, logits=True)
                        
                        dices.append(dice_score)
                        nsds.append(nsd_score)


                    # Print formatted results
                    mean_dice = np.mean(dices)
                    mean_nsd = np.mean(nsds)
                    print(f"Data Transform: {data_transform}, Dataset: {key}")
                    print(f"Mean Dice Score: {mean_dice:.4f}")
                    print(f"Mean NSD Score: {mean_nsd:.4f}\n")
                    break
