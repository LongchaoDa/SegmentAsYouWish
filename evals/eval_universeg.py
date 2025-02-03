import json
import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
import warnings
from universeg import universeg
from utils import FLanSDataset, normalized_surface_distance_np, dice_coefficient

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Suppress warnings
warnings.filterwarnings("ignore")

# Load JSON Files
with open("prompts/free_form_text_prompts.json", 'r') as file:
    label_dict = json.load(file)

with open("configs/test_data_paths.json", 'r') as file:
    test_data_path_lists = json.load(file)

with open("prompts/positional_free_form_text_prompts.json", 'r') as file:
    pos_prompt_dict = json.load(file)

with open("prompts/organ_positions.json", 'r') as file:
    pos_file_dict = json.load(file)


def get_element_appearing_more_than_twice(lst):
    """Find indices of the first element appearing at least twice in a list."""
    element_counts = Counter(lst)
    for element, count in element_counts.items():
        if count >= 2:
            return [i for i, val in enumerate(lst) if val == element][:count]
    return []


def min_max(tensor):
    """Normalize tensor to range [0,1]."""
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


if __name__ == "__main__":
    model = universeg(pretrained=True).to(device)

    for key in test_data_path_lists.keys():
        print(f"Processing dataset: {key}")

        for data_aug in [False, True]:
            # Select dataset based on augmentation flag
            dataset = FLanSDataset(
                data_path_lists={key: test_data_path_lists[key]}, 
                label_dict=label_dict, 
                data_aug=data_aug, 
                group_order=4 if data_aug else None
            )
            dataloader = DataLoader(dataset, batch_size=32, shuffle=data_aug, num_workers=4, pin_memory=True)

            dices, nsds = [], []
            for batch in dataloader:
                indices = get_element_appearing_more_than_twice(batch["label_id"].numpy())
                if not indices:
                    continue  # Skip if no valid indices found

                target_image = batch["image"][indices[0]:indices[0]+1][:,:1,::8,::8].to(device)
                target_label = batch["gt2D"][indices[0]:indices[0]+1,:,::2,::2]

                support_images = torch.cat(
                    [batch["image"][indices[i]:indices[i]+1].unsqueeze(2)[:,:1,:,::8,::8] for i in range(1, len(indices))], 
                    dim=1
                ).to(device)

                support_labels = torch.cat(
                    [batch["gt2D"][indices[1]:indices[1]+1,:,::2,::2].unsqueeze(2) for i in range(1, len(indices))], 
                    dim=1
                ).to(device)

                prediction = model(
                    min_max(target_image),        
                    min_max(support_images),      
                    min_max(support_labels),      
                ).cpu()

                dices.append(dice_coefficient(prediction, target_label))
                nsds.append(normalized_surface_distance_np(target_label.numpy(), target_label.numpy()))
                print(f"Dice: {dices[-1]:.4f}, NSD: {nsds[-1]:.4f}")

            print(f"Results for dataset: {key}, Data Augmentation: {data_aug}")
            print(f"Mean Dice: {np.mean(dices):.4f}, Mean NSD: {np.mean(nsds):.4f}\n")
