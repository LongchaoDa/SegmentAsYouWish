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
from utils import FLanSDataset, train_epoch_batch, eval_epoch_batch, train_epoch_batch_canonicalizer

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

data_path_lists = {
    "Abdomen_1K_Subtask1": "data/touse/AbdomenCT_1K/npy/CT_Abd1K/Subtask1",
    "MSD_Lung": "data/touse/MSD/npy/CT_MSD/Task06_Lung",
    "MSD_Spleen": "data/touse/MSD/npy/CT_MSD/Task09_Spleen",
    "FLARE22_train": "data/touse/FLARE22/npy/CT_Abd_train",
    "BTCV": "data/touse/BTCV/npy/CT_BTCV/Abdomen",
    "CHAOS_CT": "text_prompt/data/touse/CHAOS/npy/CT_CHAOS"
}

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

canonicalization_network = DDP(canonicalization_network.to(local_rank), device_ids=[local_rank], find_unused_parameters=True)
###### Warm Up Canonicalization Network ###### 
can_save_name = f"canonicalizer_canparams{config_yaml['canonicalization']['group_type']}_{config_yaml['canonicalization']['group_order']}"
if rank == 0:
    print(can_save_name)
    print("Warm Up Canonicalization Network...")
can_train_dataset = FLanSDataset(data_path_lists = data_path_lists, label_dict = label_dict, data_aug = False, group_order = config_yaml["canonicalization"]["group_order"])
can_train_sampler = torch.utils.data.distributed.DistributedSampler(
    can_train_dataset, num_replicas=world_size, rank=rank, shuffle=False
)
can_train_loader = DataLoader(can_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=can_train_sampler)

aug_can_train_dataset = FLanSDataset(data_path_lists = data_path_lists, label_dict = label_dict, data_aug = True, group_order = config_yaml["canonicalization"]["group_order"])
aug_can_train_sampler = torch.utils.data.distributed.DistributedSampler(
    aug_can_train_dataset, num_replicas=world_size, rank=rank, shuffle=False
)
aug_can_train_loader = DataLoader(aug_can_train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=aug_can_train_sampler)
can_optimizer = optim.Adam(canonicalization_network.parameters(), lr=0.001)

can_loss_fun = torch.nn.MSELoss()
best_can_loss = 1e6
if rank == 0:
    print(canonicalization_network.module.canonicalization_network.eqv_network[0].weights[:5].cpu().data.numpy())
    
for epoch in range(10):
    dist.barrier()
    can_loss = train_epoch_batch_canonicalizer(can_train_loader, aug_can_train_loader, canonicalization_network, can_optimizer, can_loss_fun, epoch)
    if rank == 0:
        print(canonicalization_network.module.canonicalization_network.eqv_network[0].weights[:5].cpu().data.numpy())
        print(f"Canonicalization Network Epoch {epoch}, MSE loss: {can_loss:.4f}")
        if can_loss < best_can_loss:
            best_can_loss = can_loss
            torch.save({"can_model": canonicalization_network.module.state_dict()}, join(work_dir, can_save_name + ".pth"))
            if rank == 0:
                print("new canonicalization_network checkpoint saved")

dist.destroy_process_group()



