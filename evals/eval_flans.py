import os
import sys
import argparse
# Parser setup
parser = argparse.ArgumentParser(description='Evaluate the model with specified checkpoint and GPU')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the checkpoint file')
parser.add_argument('--gpu', type=str, default='0', help='Specify GPU device')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
checkpoint_name = args.checkpoint
current_dir = os.getcwd()
# Move one level up to the parent directory
parent_dir = os.path.dirname(current_dir)
# Change the working directory to the parent directory
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
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
from torch.nn.parallel import DataParallel
import json
from modules import DiscreteGroupImageCanonicalization, ESCNNEquivariantNetwork
from modules import TextPromptEncoder, MedSAMWithCanonicalization
from modules import sam_model_registry
from utils import FLanSDataset, normalized_surface_distance_np, FLanSDataset_pos_only, test_epoch_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")




def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config_yaml = load_config("configs/train_config_main1.yaml")
# checkpoint_name = "cosine_two_stage_canonTrue_augTrue_classify_headFalse_freezeimagencoderFalse_lr_prompt0.0001_lr_imgenc0.0001_lr_imgdec0.0001_lr_can0.0001_lr_class0.0001_bz2_poolingmean_gradstraight_through_canparams9_8_3_8.pth"

work_dir = config_yaml["data"]["work_dir"]
num_epochs = config_yaml["hyperparameter"]["max_epochs"]
batch_size = config_yaml["hyperparameter"]["batch_size"]
num_workers = config_yaml["hyperparameter"]["num_workers"]
medsam_checkpoint = config_yaml["model"]["medsam_checkpoint"]
data_aug = config_yaml["train"]["data_aug"]
seed = config_yaml["hyperparameter"]["seed"]
augmented_textlabel = config_yaml["train"]["textlabel_aug"]
use_canonicalization = config_yaml["train"]["use_canonicalization"]
freeze_image_encoder = config_yaml["train"]["freeze_image_encoder"]
use_classify_head = config_yaml["train"]["classify_head"]
text_pooling = config_yaml["hyperparameter"]["text_pooling"]


# Initialize TensorBoard SummaryWriter
torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True  # To ensure deterministic behavior for certain operations
torch.backends.cudnn.benchmark = False  # Turn off optimization that may introduce randomness

# Load labels
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
    
print("Test Datasets:", test_data_path_lists.keys())
###### Initialize Text Prompt Encoder ###### 
text_prompt_encoder = TextPromptEncoder(
    embed_dim=config_yaml["model"]["text_embed_dim"],
    image_embedding_size=(64, 64),
    input_image_size = (config_yaml["data"]["image_size"], config_yaml["data"]["image_size"]),
    mask_in_chans=1,
    activation=nn.GELU,
)

sam_model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint)
medsam_prompt_encoder_state_dict = sam_model.prompt_encoder.state_dict()
## Load pretrained weights from MedSAM's prompt encoder except for the text encoder
for keys in text_prompt_encoder.state_dict().keys():
    if keys in medsam_prompt_encoder_state_dict.keys():
        text_prompt_encoder.state_dict()[keys] = deepcopy(medsam_prompt_encoder_state_dict[keys])
    else:
        assert keys.startswith("text_encoder")

###### Initialize Canonicalization Network ###### 
class CanonicalizationHyperparams:
    def __init__(self):
        self.network_type = "escnn" # group equivariant canonicalization
        self.resize_shape = config_yaml["canonicalization"]["resize_shape"] # resize shape for the canonicalization network
        self.network_hyperparams = {
            "kernel_size": config_yaml["canonicalization"]["kernel_size"], # Kernel size for the canonization network
            "hidden_dim": config_yaml["canonicalization"]["hidden_dim"],
            "out_channels": config_yaml["canonicalization"]["out_channels"], # Number of output channels for the canonization network
            "num_layers": config_yaml["canonicalization"]["num_layers"], # Number of layers in the canonization network
            "group_type": config_yaml["canonicalization"]["group_type"],#"roto-reflection", #rotation", # Type of group for the canonization network
            "group_order": config_yaml["canonicalization"]["group_order"], # Number of rotations for the canonization network 
        }
        self.beta = config_yaml["canonicalization"]["beta"]
        self.input_crop_ratio = config_yaml["canonicalization"]["input_crop_ratio"]
        self.gradient_trick = config_yaml["canonicalization"]["gradient_trick"] #"gumbel_softmax" / "straight_through"
canonicalization_hyperparams = CanonicalizationHyperparams()

# initialize the can network: 
image_shape = (config_yaml["data"]["inp_channels"], config_yaml["data"]["image_size"], config_yaml["data"]["image_size"])
canonicalization_def = ESCNNEquivariantNetwork(
    inp_channels = config_yaml["data"]["inp_channels"], 
    **canonicalization_hyperparams.network_hyperparams
)#.to(device)
canonicalization_network = DiscreteGroupImageCanonicalization(canonicalization_def, canonicalization_hyperparams, image_shape)#.to(device)


###### Initialize the Main Model ###### 
# need to adjsut the freeze option here: # todo
flan_model = MedSAMWithCanonicalization(
    image_encoder=sam_model.image_encoder,
    mask_decoder=deepcopy(sam_model.mask_decoder),
    prompt_encoder=text_prompt_encoder,
    canonicalization_network = canonicalization_network, 
    use_canonicalization = use_canonicalization,
    use_classify_head = use_classify_head,
    freeze_image_encoder=freeze_image_encoder # False
).to(device)

checkpoint = torch.load("results/foundation_model/" + checkpoint_name)
flan_model.load_state_dict(checkpoint["model"], strict = False)

###### Data Parallel training ######
# Wrap the model with DistributedDataParallel
flan_model = flan_model.to(device)
loss_funs = {
    "seg_loss": monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean'),
    "ce_loss" : nn.BCEWithLogitsLoss(reduction="mean"),
    "clf_loss" : nn.CrossEntropyLoss() if use_classify_head else None
            }

loss_coefs = {
    "seg_loss": config_yaml["hyperparameter"]["coef_seg"],
    "ce_loss" : config_yaml["hyperparameter"]["coef_ce"],
    "clf_loss" : config_yaml["hyperparameter"]["coef_clf"] if use_classify_head else None
}

#### test_epoch_batch ####

def dice_coefficient(preds, targets, smooth=0.0):
    # Apply sigmoid if using logits
    preds = torch.sigmoid(preds)

    # Convert predictions to binary (0 or 1)
    preds = (preds > 0.5).float()

    # Flatten the tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Calculate Dice Coefficient
    intersection = (preds * targets).sum()
    dice_coeff = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    return dice_coeff.item()


def test_epoch_batch_informed(eval_loader, model, loss_funs, loss_coefs, epoch = None, prompt_type = "text"):
    with torch.no_grad():
        eval_losses, seg_losses, ce_losses, clf_losses, dice_coeffs = [], [], [], [], []
        preds, trues = [], []

        # Only create a progress bar if this is the main process
        pbar = tqdm(eval_loader)
            
        for step, batch in enumerate(pbar):
            # one batch
            image, gt2D, tokens, organ_name_token = batch["image"].to(device), batch["gt2D"].to(device), batch["token"].to(device), batch["organ_name_token"].to(device)
            
            # forward pass
            # use free-form prompts
            if prompt_type == "text":
                medsam_pred, text_classification_logits = model(image, tokens)
            # use organ names
            elif prompt_type == "organ":
                medsam_pred, text_classification_logits = model(image, organ_name_token)
            preds.append(medsam_pred.cpu().data.numpy())
            trues.append(gt2D.cpu().data.numpy())

            seg_loss_value = loss_funs["seg_loss"](medsam_pred, gt2D)
            ce_loss_value = loss_funs["ce_loss"](medsam_pred, gt2D.float())
            
            if loss_funs["clf_loss"]:
                clf_loss_value = loss_funs["clf_loss"](text_classification_logits, labels - 1)

            dice_coeffs.append(dice_coefficient(medsam_pred, gt2D))
            seg_losses.append(seg_loss_value.item())
            ce_losses.append(ce_loss_value.item())
            total_loss = loss_coefs["seg_loss"] * seg_loss_value.item() + loss_coefs["ce_loss"] * ce_loss_value.item() 
            
            if loss_funs["clf_loss"]:
                clf_losses.append(clf_loss_value.item())
                total_loss += loss_coefs["clf_loss"] * clf_loss_value.item()
               
            eval_losses.append(total_loss)
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, eval total loss: {total_loss:.4f}")
        
    return np.mean(eval_losses), np.mean(seg_losses), np.mean(ce_losses), np.mean(clf_losses), np.mean(dice_coeffs), np.concatenate(preds, axis = 0), np.concatenate(trues, axis = 0)


# #### Evaluation ####
print("FLanS: Anatomy-Informed Segmentation Results (Free-Form Text)")
texts_results = []
aug_text_results = []
with torch.no_grad(): 
    for key in list(test_data_path_lists.keys()):
        test_dataset = FLanSDataset(data_path_lists = {key: test_data_path_lists[key]}, label_dict = label_dict, data_aug = False)
        test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=num_workers, pin_memory=True)
        aug_test_dataset = FLanSDataset(data_path_lists = {key: test_data_path_lists[key]}, label_dict = label_dict, data_aug = True, group_order = config_yaml["canonicalization"]["group_order"])
        aug_test_loader = DataLoader(aug_test_dataset, batch_size = 1, shuffle=False, num_workers=num_workers, pin_memory=True)
        _, _, test_ce_loss_text, _, test_dice_score_text, test_preds_text, test_trues_text = test_epoch_batch_informed(test_loader, flan_model, loss_funs, loss_coefs, prompt_type = "text")
        _, _, aug_test_ce_loss_text, _, aug_test_dice_score_text, aug_test_preds_text, aug_test_trues_text = test_epoch_batch_informed(aug_test_loader, flan_model, loss_funs, loss_coefs, prompt_type = "text")
        nsd_text = normalized_surface_distance_np(test_preds_text, test_trues_text)
        aug_nsd_text = normalized_surface_distance_np(aug_test_preds_text, aug_test_trues_text)
        print(checkpoint_name)
        print(key)
        print(f"Dice: {test_dice_score_text:.3f}, CE: {test_ce_loss_text:.3f}, NSD: {nsd_text:.3f}")
        print(f"Aug Dice: {aug_test_dice_score_text:.3f}, Aug CE: {aug_test_ce_loss_text:.3f}, Aug NSD: {aug_nsd_text:.3f}\n")
        print()
        texts_results.append([test_dice_score_text, test_ce_loss_text, nsd_text])
        aug_text_results.append([aug_test_dice_score_text, aug_test_ce_loss_text, aug_nsd_text])
    
        
torch.save({"text": texts_results,
            "aug_text": aug_text_results}, 
            "eval_scores_" + checkpoint_name[:-4] + ".pt")
        
        
#### Evaluation ####
print("FLanS: Anatomy-Informed Segmentation Results (Organ Name Only)")
organ_results = []
aug_organ_results = []
with torch.no_grad(): 
    for key in list(test_data_path_lists.keys()):
        test_dataset = FLanSDataset(data_path_lists = {key: test_data_path_lists[key]}, label_dict = label_dict, data_aug = False)
        test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=num_workers, pin_memory=True)
        aug_test_dataset = FLanSDataset(data_path_lists = {key: test_data_path_lists[key]}, label_dict = label_dict, data_aug = True, group_order = config_yaml["canonicalization"]["group_order"])
        aug_test_loader = DataLoader(aug_test_dataset, batch_size = 1, shuffle=False, num_workers=num_workers, pin_memory=True)
        _, _, test_ce_loss_organ, _, test_dice_score_organ, test_preds_organ, test_trues_organ = test_epoch_batch_informed(test_loader, flan_model, loss_funs, loss_coefs, prompt_type = "organ")
        _, _, aug_test_ce_loss_organ, _, aug_test_dice_score_organ, aug_test_preds_organ, aug_test_trues_organ = test_epoch_batch_informed(aug_test_loader, flan_model, loss_funs, loss_coefs, prompt_type = "organ")
        nsd_organ = normalized_surface_distance_np(test_preds_organ, test_trues_organ)
        aug_nsd_organ = normalized_surface_distance_np(aug_test_preds_organ, aug_test_trues_organ)
        print(checkpoint_name)
        print(key)
        print(f"Dice: {test_dice_score_organ:.3f}, CE: {test_ce_loss_organ:.3f}, NSD: {nsd_organ:.3f}")
        print(f"Aug Dice: {aug_test_dice_score_organ:.3f}, Aug CE: {aug_test_ce_loss_organ:.3f}, Aug NSD: {aug_nsd_organ:.3f}\n")
        print()
        organ_results.append([test_dice_score_organ, test_ce_loss_organ, nsd_organ])
        aug_organ_results.append([aug_test_dice_score_organ, aug_test_ce_loss_organ, aug_nsd_organ])
        
torch.save({"text": texts_results,
            "aug_text": aug_text_results, 
            "organ": organ_results,
            "aug_organ": aug_organ_results},
            "eval_scores_" + checkpoint_name[:-4] + ".pt")

agnostic_results = []
## please turn off canonicalization when evaluating on anatomy-agnostic prompts as we only have gts for the original images.
flan_model.use_canonicalization = False
print("FLanS: Anatomy-Agnostic Segmentation Results") 
with torch.no_grad(): 
    for key in list(test_data_path_lists.keys()):
        test_dataset = FLanSDataset_pos_only(data_path_lists = {key: test_data_path_lists[key]},
                                                                  label_dict = label_dict, 
                                                                  pos_file_dict = pos_file_dict, 
                                                                  pos_prompt_dict = pos_prompt_dict,
                                                                  pos_box_dict = pos_box_dict)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)
        _, test_seg_loss_agnostic, test_ce_loss_agnostic, _, test_dice_score_agnostic, test_preds_agnostic, test_trues_agnostic = test_epoch_batch(test_loader, flan_model, loss_funs, loss_coefs)
        nsd_agnostic = normalized_surface_distance_np(test_preds_agnostic, test_trues_agnostic)
        print(checkpoint_name)
        print(key)
        print(f"Dice: {test_dice_score_agnostic:.3f}, CE: {test_ce_loss_agnostic:.3f}, NSD: {nsd_agnostic:.3f}")
        print()
        agnostic_results.append([test_dice_score_agnostic, test_ce_loss_agnostic, nsd_agnostic])
        
torch.save({"text": texts_results,
            "aug_text": aug_text_results, 
            "organ": organ_results,
            "aug_organ": aug_organ_results,
            "agnostic": agnostic_results},
            "eval_scores_" + checkpoint_name[:-4] + ".pt")

