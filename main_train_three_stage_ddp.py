import os
import sys
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
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from transformers import CLIPTokenizer, CLIPTextModel
import argparse
from torch.nn.parallel import DataParallel
import json
from torch.utils.tensorboard import SummaryWriter

from modules import DiscreteGroupImageCanonicalization, ESCNNEquivariantNetwork
from modules import TextPromptEncoder, MedSAMWithCanonicalization
from modules import sam_model_registry
from utils import FLanSDataset, train_epoch_batch, eval_epoch_batch, train_epoch_batch_canonicalizer, FLanSDataset_pos, FLanSDataset_pos_only

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the process group
dist.init_process_group(backend='nccl')

# Get local rank from the environment variable
local_rank = int(os.environ["LOCAL_RANK"])

# Set the device for this process
torch.cuda.set_device(local_rank)

# Get the rank and world size
rank = dist.get_rank()
world_size = dist.get_world_size()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings('ignore')


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_args():
    parser = argparse.ArgumentParser(description='Training a model with config')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    return parser.parse_args()

args = setup_args()

# Load YAML configuration
config_yaml = load_config(args.config)


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
makedirs(work_dir, exist_ok=True)
writer = SummaryWriter(work_dir + "/log")
torch.cuda.empty_cache()
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Load labels
with open(config_yaml["data"]["prompt_path"], 'r') as file:
    label_dict = json.load(file)

with open("configs/train_data_paths.json", 'r') as file:
    train_data_path_lists = json.load(file)

with open("configs/test_data_paths.json", 'r') as file:
    test_data_path_lists = json.load(file)
      
with open("configs/train_pos_data_paths.json", 'r') as file:
    pos_data_path_lists = json.load(file)
    
with open("prompts/positional_free_form_text_prompts.json", 'r') as file:
    pos_prompt_dict = json.load(file)

with open("prompts/organ_positions.json", 'r') as file:
    pos_file_dict = json.load(file)

with open("prompts/organ_bounding_boxes.json", 'r') as file:
    pos_box_dict = json.load(file)
    
if rank == 0:
    print("Train Datasets:", train_data_path_lists.keys())
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
can_save_name = f"canonicalizer_canparams{config_yaml['canonicalization']['group_type']}_{config_yaml['canonicalization']['group_order']}"
if use_canonicalization:
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
else:
    canonicalization_network = None
    
###### Stage One: Warm Up Canonicalization Network ###### 

# load pretrained d8 canonicalizer
canonicalization_network.load_state_dict(torch.load("results/foundation_model/canonicalizer_canparamsroto-reflection_8.pth")["can_model"])
if rank == 0:
    print("Canonicalization Network Loaded")

# if rank == 0:
#     print("Warm Up Canonicalization Network")
# can_train_dataset = NpyDataset(data_root=config_yaml["data"]["train_npy_path"], data_aug=False, label_dict = label_dict)
# can_train_sampler = torch.utils.data.distributed.DistributedSampler(
#     can_train_dataset, num_replicas=world_size, rank=rank, shuffle=False
# )
# can_train_loader = DataLoader(can_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=can_train_sampler)

# aug_can_train_dataset = NpyDataset(data_root=config_yaml["data"]["train_npy_path"], data_aug=True, label_dict = label_dict)
# aug_can_train_sampler = torch.utils.data.distributed.DistributedSampler(
#     aug_can_train_dataset, num_replicas=world_size, rank=rank, shuffle=False
# )
# aug_can_train_loader = DataLoader(aug_can_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=aug_can_train_sampler)
# can_optimizer = optim.Adam(canonicalization_network.parameters(), lr=0.005)

# can_save_name = f"canonicalizer_canparams{config_yaml['canonicalization']['group_type']}_{config_yaml['canonicalization']['group_order']}"


# can_loss_fun = torch.nn.MSELoss()
# if rank == 0:
#     print(canonicalization_network.module.canonicalization_network.eqv_network[0].weights[:5].cpu().data.numpy())
# for epoch in range(10):
#     dist.barrier()
#     can_loss = train_epoch_batch_canonicalizer(can_train_loader, aug_can_train_loader, canonicalization_network, can_optimizer, can_loss_fun, epoch)
#     if rank == 0:
#         print(canonicalization_network.module.canonicalization_network.eqv_network[0].weights[:5].cpu().data.numpy())
#         print(f"Canonicalization Network Epoch {epoch}, MSE loss: {can_loss:.4f}")
#         if can_loss < best_can_loss:
#             best_can_loss = can_loss
#             torch.save({"can_model": canonicalization_network.module.state_dict()}, join(work_dir, can_save_name + ".pth"))



###### Initialize the Main Model ###### 
# need to adjsut the freeze option here: # todo
flans_model = MedSAMWithCanonicalization(
    image_encoder=sam_model.image_encoder,
    mask_decoder=deepcopy(sam_model.mask_decoder),
    prompt_encoder=text_prompt_encoder,
    canonicalization_network = canonicalization_network,  # 等变网络
    use_canonicalization = use_canonicalization,
    use_classify_head = use_classify_head,
    freeze_image_encoder=freeze_image_encoder # False
).to(local_rank)
if rank == 0:
    print(f"FLanS size: {sum(p.numel() for p in flans_model.parameters())/10e6} M" )


###### Define Optimizer ######
optim_params = []
if config_yaml["hyperparameter"]["learning_rate"]["prompt_encoder_lr"] > 0.0:
    optim_params.append({'params': flans_model.prompt_encoder.text_encoder_head.parameters(), 'lr': config_yaml["hyperparameter"]["learning_rate"]["prompt_encoder_lr"]})
if config_yaml["hyperparameter"]["learning_rate"]["image_encoder_lr"] > 0.0 or not freeze_image_encoder:
    optim_params.append({'params': flans_model.image_encoder.parameters(), 'lr': config_yaml["hyperparameter"]["learning_rate"]["image_encoder_lr"]})
if config_yaml["hyperparameter"]["learning_rate"]["mask_decoder_lr"] > 0.0:
    optim_params.append({'params': flans_model.mask_decoder.parameters(), 'lr': config_yaml["hyperparameter"]["learning_rate"]["mask_decoder_lr"]})
if config_yaml["hyperparameter"]["learning_rate"]["text_classification_head_lr"] > 0.0 and use_classify_head:
    optim_params.append({'params': flans_model.text_classification_head.parameters(), 'lr': config_yaml["hyperparameter"]["learning_rate"]["text_classification_head_lr"]})
# if config_yaml["hyperparameter"]["learning_rate"]["canonicalization_network_lr"] > 0.0 and use_canonicalization:
#     optim_params.append({'params': flans_model.canonicalization_network.parameters(), 'lr': config_yaml["hyperparameter"]["learning_rate"]["canonicalization_network_lr"]})

###### Data Parallel training ######
# Wrap the model with DistributedDataParallel
flans_model = DDP(flans_model, device_ids=[local_rank], find_unused_parameters=True)

optimizer = optim.AdamW(optim_params)
if rank == 0:
    print(f'Number of parameters to update: {sum(p.numel() for p in flans_model.parameters() if p.requires_grad)/1e6} M')

# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = config_yaml["hyperparameter"]["decay_rate"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

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

# Set up the distributed sampler
train_dataset = FLanSDataset_pos(data_path_lists = train_data_path_lists, label_dict = label_dict, pos_file_dict = pos_file_dict, pos_prompt_dict = pos_prompt_dict, data_aug = False, group_order = config_yaml["canonicalization"]["group_order"])
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank, shuffle=True
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=train_sampler)


train_dataset_pos = FLanSDataset_pos_only(data_path_lists = pos_data_path_lists, 
                                          label_dict = label_dict, 
                                          pos_file_dict = pos_file_dict, 
                                          pos_prompt_dict = pos_prompt_dict,
                                          pos_box_dict = pos_box_dict)
train_sampler_pos = torch.utils.data.distributed.DistributedSampler(
    train_dataset_pos, num_replicas=world_size, rank=rank, shuffle=True
)
train_loader_pos = DataLoader(train_dataset_pos, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=train_sampler_pos)


if data_aug:
    aug_train_dataset = FLanSDataset(data_path_lists = train_data_path_lists, label_dict = label_dict, data_aug = True, group_order = config_yaml["canonicalization"]["group_order"])
    aug_train_sampler = torch.utils.data.distributed.DistributedSampler(
        aug_train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    aug_train_loader = DataLoader(aug_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=aug_train_sampler)

eval_dataset = FLanSDataset(data_path_lists = test_data_path_lists, label_dict = label_dict, data_aug = True, group_order = config_yaml["canonicalization"]["group_order"])
eval_sampler = torch.utils.data.distributed.DistributedSampler(
    eval_dataset, num_replicas=world_size, rank=rank, shuffle=False
)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=eval_sampler)

                                         
test_dataset = FLanSDataset(data_path_lists = test_data_path_lists, label_dict = label_dict, data_aug = False, group_order = config_yaml["canonicalization"]["group_order"])
test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_dataset, num_replicas=world_size, rank=rank, shuffle=False
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=test_sampler)

aug_test_dataset = FLanSDataset(data_path_lists = test_data_path_lists, label_dict = label_dict, data_aug = True, group_order = config_yaml["canonicalization"]["group_order"])
aug_test_sampler = torch.utils.data.distributed.DistributedSampler(
    aug_test_dataset, num_replicas=world_size, rank=rank, shuffle=False
)
aug_test_loader = DataLoader(aug_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=aug_test_sampler)


# handle distribute process
# For loading checkpoints, all processes should load the model weights
if config_yaml["model"]["resume"]:
    checkpoint = torch.load("results/foundation_model2/48_checkpoint_cosine_two_stage_canonTrue_augTrue_classify_headFalse_freezeimagencoderFalse_lr_prompt0.0001_lr_imgenc0.0001_lr_imgdec0.0001_lr_can0.0001_lr_class0.0001_bz2_poolingmean_gradstraight_through_canparams9_8_3_8.pth", map_location='cuda:{}'.format(local_rank))
    flans_model.module.load_state_dict(checkpoint["model"], strict = False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    if rank == 0:
        print(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}")
else:
    start_epoch = 0
    best_loss = 1e10


# Create a unique name for the final results file
save_name = (
    f"cosine_"
    f"two_stage_"
    f"canon{use_canonicalization}_"
    f"aug{data_aug}_"
    f"classify_head{use_classify_head}_"
    f"freezeimagencoder{freeze_image_encoder}_"
    f"lr_prompt{config_yaml['hyperparameter']['learning_rate']['prompt_encoder_lr']}_"
    f"lr_imgenc{config_yaml['hyperparameter']['learning_rate']['image_encoder_lr']}_"
    f"lr_imgdec{config_yaml['hyperparameter']['learning_rate']['mask_decoder_lr']}_"
    f"lr_can{config_yaml['hyperparameter']['learning_rate']['canonicalization_network_lr']}_"
    f"lr_class{config_yaml['hyperparameter']['learning_rate']['text_classification_head_lr']}_"
    f"bz{config_yaml['hyperparameter']['batch_size']}_"
    f"pooling{config_yaml['hyperparameter']['text_pooling']}_"
    f"grad{config_yaml['canonicalization']['gradient_trick']}_"
    f"canparams{config_yaml['canonicalization']['kernel_size']}_{config_yaml['canonicalization']['hidden_dim']}_{config_yaml['canonicalization']['num_layers']}_{config_yaml['canonicalization']['group_order']}"
)

if rank == 0:
    print("mode name:", save_name)
####################################################################
print()
###### Stage Two: train all modules except canonilizatin network ######
if use_canonicalization:
    if rank == 0:
        print("###Stage One###: train all modules except canonilizatin network on the Non-transformed training data.")
    flans_model.module.use_canonicalization = False # disable canonilizatin network for noe
    
epoch_train_losses = []
flans_model.train()
for epoch in range(start_epoch, config_yaml["hyperparameter"]["stage_one_epochs"]):
    torch.cuda.empty_cache()

    train_total_loss, losses = train_epoch_batch(train_loader_pos, flans_model, optimizer, loss_funs, loss_coefs, epoch)
    train_total_loss, losses = train_epoch_batch(train_loader, flans_model, optimizer, loss_funs, loss_coefs, epoch)
    epoch_train_losses.append(train_total_loss)
    scheduler.step()
    # Gather results across all processes
    dist.barrier()
    if rank == 0:
        if train_total_loss < best_loss:
            print(f"New best train loss: {best_loss:.4f} -> {train_total_loss:.4f}")
            best_loss = train_total_loss
            best_model = flans_model
            if rank == 0:
                checkpoint = {
                    "model": flans_model.module.state_dict(),
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss
                }
                torch.save(checkpoint, join(work_dir, save_name + ".pth"))
        torch.save(checkpoint, join(work_dir, str(epoch) + "_checkpoint_" + save_name + ".pth"))
        
print()

###### Stage Three: train canonilizatin network and segmentation network on the augmented data. ######
if data_aug:
    if rank == 0:
        print("###Stage Two###: train canonilizatin network on the Transformed training data.")
    
    epoch_train_losses = []
    epoch_eval_losses = []
    epoch_time = []
    epoch_no_improve = 0
    if not config_yaml["model"]["resume"]:
        best_loss = 1e10
    
    if use_canonicalization:
        canonicalization_params = {'params': flans_model.module.canonicalization_network.parameters(), 'lr': config_yaml["hyperparameter"]["learning_rate"]["canonicalization_network_lr"]}
        optimizer.add_param_group(canonicalization_params)
        flans_model.module.use_canonicalization = True
        
        if rank == 0:
            print(f"Optimizing canonicalization_network as well")

    for epoch in range(51, num_epochs):#
        
        torch.cuda.empty_cache()
        epoch_start_time = time()
        flans_model.train()
        flans_model.module.use_canonicalization = False
        _, _ = train_epoch_batch(train_loader_pos, flans_model, optimizer, loss_funs, loss_coefs, epoch)
        
        flans_model.module.use_canonicalization = True
        train_total_loss, losses = train_epoch_batch(aug_train_loader, flans_model, optimizer, loss_funs, loss_coefs, epoch)
        epoch_train_losses.append(train_total_loss)

        scheduler.step()
        flans_model.eval()
        eval_total_loss, eval_seg_loss, eval_ce_loss, eval_clf_loss, eval_dice_score, _, _ = eval_epoch_batch(eval_loader, flans_model, loss_funs, loss_coefs, epoch)
        epoch_eval_losses.append(eval_total_loss)

        epoch_end_time = time()
        epoch_time.append(epoch_end_time - epoch_start_time)

        dist.barrier()
        
        if rank == 0:
            # Save the model if rank 0
            if eval_total_loss < best_loss:
                print(f"New best eval loss: {best_loss:.4f} -> {eval_total_loss:.4f}")
                best_loss = eval_total_loss
                best_model = flans_model
                # parallel running:
                checkpoint = {
                    "model": flans_model.module.state_dict() if torch.cuda.device_count() > 1 else flans_model.state_dict(),
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss
                }
                torch.save(checkpoint, join(work_dir, save_name + "_2.pth"))
                epoch_no_improve = 0
            else:
                epoch_no_improve += 1
            torch.save(checkpoint, join(work_dir, str(epoch) + "_checkpoint_" + save_name + "_2.pth"))

        if len(epoch_train_losses) > 50 and epoch_no_improve > config_yaml["hyperparameter"]["patience"]:
            break   
    

if rank == 0:
    with torch.no_grad(): 
        print("Evaluating the best model on the test sets")
        test_total_loss, test_seg_loss, test_ce_loss, test_clf_loss, test_dice_score, test_preds, test_trues = eval_epoch_batch(test_loader, best_model, loss_funs, loss_coefs)
        aug_test_total_loss, aug_test_seg_loss, aug_test_ce_loss, aug_test_clf_loss, aug_test_dice_score, aug_test_preds, aug_test_trues = eval_epoch_batch(aug_test_loader, best_model, loss_funs, loss_coefs)
        torch.save({
            "model": best_model.module.state_dict() if torch.cuda.device_count() > 1 else best_model.state_dict(),
            "test_preds": test_preds,
            "test_trues": test_trues, 
            "test_dice_score": test_dice_score,
            "test_total_loss": test_total_loss, 
            "test_seg_loss": test_seg_loss, 
            "test_ce_loss": test_ce_loss,
            "test_clf_loss": test_clf_loss,
            "aug_test_preds": aug_test_preds,
            "aug_test_trues": aug_test_trues, 
            "aug_test_dice_score": aug_test_dice_score,
            "aug_test_total_loss": aug_test_total_loss, 
            "aug_test_seg_loss": aug_test_seg_loss, 
            "aug_test_ce_loss": aug_test_ce_loss,
            "aug_test_clf_loss": aug_test_clf_loss,
            }, join(work_dir, save_name + ".pt"))
    
dist.destroy_process_group()
writer.close()

