import json
import glob
import os
from os.path import join, basename
import torch
from tqdm import tqdm
import cv2
import random
import numpy as np
import torch
from datetime import datetime
from transformers import CLIPTokenizer
from .group_utils import apply_d4_transform, apply_d8_transform
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


import torch.distributed as dist

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def save_labels(save_path=None):
    if save_path == None:
        save_path = "/home/ec2-user/textSegment/Proposal/MedSAM/extensions/text_prompt"
    label_dict = {
        1: ["Liver", "liver"],
        2: ["Right Kidney", "right kidney", "kidney"],
        3: ["Spleen", "spleen"],
        4: ["Pancreas", "pancreas"],
        5: ["Aorta", "aorta"],
        6: ["Inferior Vena Cava", "IVC", "inferior vena cava", "ivc", "vena cava", "vena", "cava"],
        7: ["Right Adrenal Gland", "RAG", "right adrenal gland", "rag", "adrenal gland", "adrenal"],
        8: ["Left Adrenal Gland", "LAG", "left adrenal gland", "lag", "adrenal gland", "adrenal"],
        9: ["Gallbladder", "gallbladder"],
        10: ["Esophagus", "esophagus"],
        11: ["Stomach", "stomach"],
        12: ["Duodenum", "duodenum"],
        13: ["Left Kidney", "left kidney", "kidney"],
    }

    # Write the dictionary to a JSON file
    with open(save_path+'organ_labels.json', 'w') as file:
        json.dump(label_dict, file, indent=4)
        
        
        
###### Main Dataset class but without anatomy agnostic prompts ######
class FLanSDataset(Dataset): 
    def __init__(self, 
                 data_path_lists, # a dictionary of datasets' paths 
                 label_dict = None, # free-form text prompts
                 image_size=1024, # input image resolution
                 gt_size = 256, # mask resolution
                 data_aug=False, # whether to perform data augmentation
                 group_order = 4 # group order
                ):

        self.data_path_lists = data_path_lists
        self.img_path = {}
        self.gt_path = {}
        self.gt_path_files = {}
        self.sample_count = []
        self.data_names = []
        self.group_order = group_order
        
        for key in self.data_path_lists:
            path = self.data_path_lists[key]
            self.img_path[key] = join(path, 'imgs')
            self.gt_path[key] = join(path, 'gts')
            self.gt_path_files[key] = sorted(glob.glob(join(self.gt_path[key], '**/*.npy'), recursive=True))
            self.gt_path_files[key] = [file for file in self.gt_path_files[key] if os.path.isfile(join(self.img_path[key], os.path.basename(file)))]
            self.sample_count.append(len(self.gt_path_files[key]))
            self.data_names.append(key)
            
        self.image_size = image_size
        self.gt_size = gt_size
        self.data_aug = data_aug
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")   
        self.label_dict = label_dict

    def __len__(self):
        return sum(self.sample_count)

    def __getitem__(self, index):
        
        dataset_name, sample_index = self.find_dataset_and_sample(index)
        img_name = basename(self.gt_path_files[dataset_name][sample_index])
        # print(index, dataset_name, sample_index)
        
        ### Preprocess Input Images ###
        img_1024 = np.load(join(self.img_path[dataset_name], img_name), 'r', allow_pickle=True) # (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1)) 
        if img_1024.shape[0] == 1:
            img_1024 = img_1024.repeat(3, 1, 1)
            
        assert img_1024.shape[0] == 3, 'input image should have 3 channels'
        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0, 'image should be normalized to [0, 1]'
        
        ### Preprocess Ground Truth Masks ###
        gt = np.load(self.gt_path_files[dataset_name][sample_index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        if gt.shape[0] != 256 or gt.shape[1] != 256:
            ## To match the shape of low_res_masks
            gt_resize = cv2.resize(
                gt,
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            gt_resize = gt.astype(np.uint8)
    
        label_ids = np.unique(gt_resize)[1:]
        label_id = random.choice(label_ids.tolist())
        try:
            gt2D = np.uint8(gt_resize == label_id) # only one label, (256, 256)
        except:
            label_id = np.max(gt)
            gt2D = np.uint8(gt_resize == label_id) # only one label, (256, 256)

        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            # print("perform data aug")
            if self.group_order == 4:
                img_1024, gt2D = apply_d4_transform(img_1024, np.expand_dims(gt2D, axis=0))
                gt2D = np.squeeze(gt2D, axis=0)
            elif self.group_order == 8:
                img_1024, gt2D = apply_d8_transform(img_1024, np.expand_dims(gt2D, axis=0))
                gt2D = np.squeeze(gt2D, axis=0)
            else:
                raise NotImplementedError("group order cannot be bigger than 8")

            
        gt2D = np.uint8(gt2D > 0)

        ## Ramdonly select a synonum of the label
        caption = random.choice(list(self.label_dict[dataset_name][str(label_id)].values())[0])
        text_token = self.tokenize_text(caption)
        
        organ_name = list(self.label_dict[dataset_name][str(label_id)].keys())
        organ_name_token = self.tokenize_text(organ_name)

        return {
            "image": torch.from_numpy(img_1024.copy().copy()).float(),
            "gt2D": torch.from_numpy(gt2D[None, :,:].copy()).long(),
            "text": [caption],
            "token": text_token,
            "image_name": img_name,
            "label_id": label_id,
            "organ_name": organ_name,
            "organ_name_token": organ_name_token
        }

    def find_dataset_and_sample(self, sampled_number):
        # Find the dataset index where the sampled number belongs
        cumulative_samples = np.cumsum(self.sample_count)
        for i, cumulative in enumerate(cumulative_samples):
            if sampled_number < cumulative:
                # The dataset is found, calculate the sample index within this dataset
                dataset_index = i
                if i == 0:
                    sample_index = sampled_number
                else:
                    sample_index = sampled_number - cumulative_samples[i-1]
                return self.data_names[dataset_index], sample_index
            
    def tokenize_text(self, text):
        """
        Tokenize text using CLIP tokenizer
        """
        return self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
        ).input_ids.squeeze(0)


###### Dataset class with both anatomy informed and anatomy agnostic prompts sampled ######
class FLanSDataset_pos(Dataset): 
    def __init__(self, 
                 data_path_lists, # a dictionary of datasets' paths 
                 label_dict = None, # free-form anatomy informed text prompts
                 image_size=1024, # input image resolution
                 gt_size = 256, # mask resolution
                 data_aug=False, # whether to perform data augmentation
                 group_order = 4, # group order
                 pos_file_dict = None, # files used to generate anatomy agnostic prompts
                 pos_prompt_dict = None, # anatomy agnostic prompts
                 ):

        self.data_path_lists = data_path_lists
        self.img_path = {}
        self.gt_path = {}
        self.gt_path_files = {}
        self.sample_count = []
        self.data_names = []
        self.group_order = group_order
        self.pos_prompt_dict = pos_prompt_dict
        self.pos_file_dict = pos_file_dict
        
        for key in self.data_path_lists:
            path = self.data_path_lists[key]
            self.img_path[key] = join(path, 'imgs')
            self.gt_path[key] = join(path, 'gts')
            self.gt_path_files[key] = sorted(glob.glob(join(self.gt_path[key], '**/*.npy'), recursive=True))
            self.gt_path_files[key] = [file for file in self.gt_path_files[key] if os.path.isfile(join(self.img_path[key], os.path.basename(file)))]
            self.sample_count.append(len(self.gt_path_files[key]))
            self.data_names.append(key)
            
        self.image_size = image_size
        self.gt_size = gt_size
        self.data_aug = data_aug
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")   
        self.label_dict = label_dict

    def __len__(self):
        return sum(self.sample_count)

    def __getitem__(self, index):
        
        dataset_name, sample_index = self.find_dataset_and_sample(index)
        img_name = basename(self.gt_path_files[dataset_name][sample_index])
        # print(index, dataset_name, sample_index)
        
        ### Preprocess Input Images ###
        img_1024 = np.load(join(self.img_path[dataset_name], img_name), 'r', allow_pickle=True) # (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1)) 
        if img_1024.shape[0] == 1:
            img_1024 = img_1024.repeat(3, 1, 1)
            
        assert img_1024.shape[0] == 3, 'input image should have 3 channels'
        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0, 'image should be normalized to [0, 1]'
        
        ### Preprocess Ground Truth Masks ###
        gt = np.load(self.gt_path_files[dataset_name][sample_index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        if gt.shape[0] != 256 or gt.shape[1] != 256:
            ## To match the shape of low_res_masks
            gt_resize = cv2.resize(
                gt,
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            gt_resize = gt.astype(np.uint8)
    
        label_ids = np.unique(gt_resize)[1:]
        label_id = random.choice(label_ids.tolist())
        try:
            gt2D = np.uint8(gt_resize == label_id) # only one label, (256, 256)
        except:
            label_id = np.max(gt)
            gt2D = np.uint8(gt_resize == label_id) # only one label, (256, 256)

        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            # print("perform data aug")
            if self.group_order == 4:
                img_1024, gt2D = apply_d4_transform(img_1024, np.expand_dims(gt2D, axis=0))
                gt2D = np.squeeze(gt2D, axis=0)
            elif self.group_order == 8:
                img_1024, gt2D = apply_d8_transform(img_1024, np.expand_dims(gt2D, axis=0))
                gt2D = np.squeeze(gt2D, axis=0)
            else:
                raise NotImplementedError("group order cannot be bigger than 8")

            
        gt2D = np.uint8(gt2D > 0)

        # # and random.random() <= 0.3
        # if self.pos_file_dict is not None and dataset_name in self.pos_file_dict.keys() and img_name in self.pos_file_dict[dataset_name].keys():
        #     print("find an image", label_id, self.pos_file_dict[dataset_name][img_name].keys())
            
        if random.random() <= 0.8 and self.pos_file_dict is not None and dataset_name in self.pos_file_dict.keys() and img_name in self.pos_file_dict[dataset_name].keys() and str(label_id) in self.pos_file_dict[dataset_name][img_name].keys():
            caption = random.choice(self.pos_prompt_dict[random.choice(self.pos_file_dict[dataset_name][img_name][str(label_id)])])
            # print("added a positional prompt")
        else:
            ## Ramdonly select a synonum of the label
            caption = random.choice(list(self.label_dict[dataset_name][str(label_id)].values())[0])
        text_token = self.tokenize_text(caption)

        return {
            "image": torch.from_numpy(img_1024.copy().copy()).float(),
            "gt2D": torch.from_numpy(gt2D[None, :,:].copy()).long(),
            "text": [caption],
            "token": text_token,
            "image_name": img_name,
            "label": label_id
        }

    def find_dataset_and_sample(self, sampled_number):
        # Find the dataset index where the sampled number belongs
        cumulative_samples = np.cumsum(self.sample_count)
        for i, cumulative in enumerate(cumulative_samples):
            if sampled_number < cumulative:
                # The dataset is found, calculate the sample index within this dataset
                dataset_index = i
                if i == 0:
                    sample_index = sampled_number
                else:
                    sample_index = sampled_number - cumulative_samples[i-1]
                return self.data_names[dataset_index], sample_index
            
    def tokenize_text(self, text):
        """
        Tokenize text using CLIP tokenizer
        """
        return self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
        ).input_ids.squeeze(0)


###### Dataset class only with anatomy agnostic prompts sampled ######
class FLanSDataset_pos_only(Dataset): 
    def __init__(self, 
                 data_path_lists,  # a dictionary of datasets' paths 
                 label_dict = None, # free-form anatomy informed text prompts
                 image_size=1024,
                 gt_size = 256, 
                 pos_file_dict = None, # organs' positions
                 pos_prompt_dict = None, # files used to generate anatomy agnostic prompts
                 pos_box_dict = None, # anatomy agnostic prompts
                 ):

        self.data_path_lists = data_path_lists
        self.img_path = {}
        self.gt_path = {}
        self.gt_path_files = {}
        self.sample_count = []
        self.data_names = []
        self.pos_prompt_dict = pos_prompt_dict
        self.pos_file_dict = pos_file_dict
        self.pos_box_dict = pos_box_dict
        
        for key in self.data_path_lists:
            path = self.data_path_lists[key]
            self.img_path[key] = join(path, 'imgs')
            self.gt_path[key] = join(path, 'gts')
            self.gt_path_files[key] = sorted(glob.glob(join(self.gt_path[key], '**/*.npy'), recursive=True))
            self.gt_path_files[key] = [file for file in self.gt_path_files[key] if os.path.isfile(join(self.img_path[key], os.path.basename(file))) and 
                                       basename(file) in self.pos_file_dict[key].keys()]
            self.sample_count.append(len(self.gt_path_files[key]))
            self.data_names.append(key)
            
        self.image_size = image_size
        self.gt_size = gt_size
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")   
        self.label_dict = label_dict

    def __len__(self):
        return sum(self.sample_count)

    def __getitem__(self, index):
        
        dataset_name, sample_index = self.find_dataset_and_sample(index)
        img_name = basename(self.gt_path_files[dataset_name][sample_index])
        # print(index, dataset_name, sample_index)
        
        ### Preprocess Input Images ###
        img_1024 = np.load(join(self.img_path[dataset_name], img_name), 'r', allow_pickle=True) # (H, W, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1)) 
        if img_1024.shape[0] == 1:
            img_1024 = img_1024.repeat(3, 1, 1)
            
        assert img_1024.shape[0] == 3, 'input image should have 3 channels'
        assert np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0, 'image should be normalized to [0, 1]'
        
        ### Preprocess Ground Truth Masks ###
        gt = np.load(self.gt_path_files[dataset_name][sample_index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        if gt.shape[0] != 256 or gt.shape[1] != 256:
            ## To match the shape of low_res_masks
            gt_resize = cv2.resize(
                gt,
                (256, 256),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.uint8)
        else:
            gt_resize = gt.astype(np.uint8)
    
        # label_ids = np.unique(gt_resize)[1:]
        label_id = random.choice(list(self.pos_file_dict[dataset_name][img_name].keys()))
        gt2D = np.uint8(gt_resize == int(label_id)) 

        gt2D = np.uint8(gt2D > 0)

        caption = random.choice(self.pos_prompt_dict[random.choice(self.pos_file_dict[dataset_name][img_name][str(label_id)])])
        box = torch.tensor(self.pos_box_dict[dataset_name][img_name][str(label_id)])

        text_token = self.tokenize_text(caption)

        return {
            "image": torch.from_numpy(img_1024.copy().copy()).float(),
            "gt2D": torch.from_numpy(gt2D[None, :,:].copy()).long(),
            "text": [caption],
            "box": box,
            "token": text_token,
            "image_name": img_name,
            "label_id": label_id
        }

    def find_dataset_and_sample(self, sampled_number):
        # Find the dataset index where the sampled number belongs
        cumulative_samples = np.cumsum(self.sample_count)
        for i, cumulative in enumerate(cumulative_samples):
            if sampled_number < cumulative:
                # The dataset is found, calculate the sample index within this dataset
                dataset_index = i
                if i == 0:
                    sample_index = sampled_number
                else:
                    sample_index = sampled_number - cumulative_samples[i-1]
                return self.data_names[dataset_index], sample_index
            
    def tokenize_text(self, text):
        """
        Tokenize text using CLIP tokenizer
        """
        return self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt" 
        ).input_ids.squeeze(0)


    
            
def train_epoch_batch_canonicalizer(can_train_loader, aug_can_train_loader, can_model, can_optimizer, can_loss_fun, epoch = None):
    train_losses = []
    
    rank = dist.get_rank() if is_distributed() else 0
    
    if rank == 0:
        pbar = tqdm(zip(can_train_loader, aug_can_train_loader))
    else:
        pbar = zip(can_train_loader, aug_can_train_loader)

    can_losses = []
    for step, (orig_batch, trans_batch) in enumerate(pbar):
        can_optimizer.zero_grad()
        orig_image = orig_batch["image"].to(device)
        tran_image = trans_batch["image"].to(device)
        out = can_model(orig_image)
        can_loss = can_loss_fun(tran_image, out)
        prior_loss = can_model.module.get_prior_regularization_loss()
        can_loss += prior_loss * 100
        can_losses.append(can_loss.item())
        can_loss.backward()
        can_optimizer.step()
        if rank == 0:
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, train total loss: {can_loss.item():.4f}")

    return np.mean(can_losses)


def train_epoch_batch(train_loader, model, optimizer, loss_funs, loss_coefs, epoch = None):
    train_losses = []
    
    rank = dist.get_rank() if is_distributed() else 0
    
    if rank == 0:
        pbar = tqdm(train_loader)
    else:
        pbar = train_loader

    for step, batch in enumerate(pbar):
        optimizer.zero_grad()
        # one batch
        # image, gt2D, tokens, labels = batch["image"].to(device), batch["gt2D"].to(device), batch["token"].to(device), batch["label_id"].to(device)
        image, gt2D, tokens = batch["image"].to(device), batch["gt2D"].to(device), batch["token"].to(device)
        # forward pass
        medsam_pred, text_classification_logits = model(image, tokens)
        # print(medsam_pred.shape, gt2D.shape, text_classification_logits.shape)
        seg_loss_value = loss_funs["seg_loss"](medsam_pred, gt2D)
        loss = loss_coefs["seg_loss"] * seg_loss_value + loss_coefs["ce_loss"] * loss_funs["ce_loss"](medsam_pred, gt2D.float())
        if model.module.use_canonicalization:
            prior_loss = model.module.canonicalization_network.get_prior_regularization_loss()
            loss += prior_loss
        if loss_funs["clf_loss"]:
            classification_loss_value = loss_coefs["clf_loss"] * loss_funs["clf_loss"](text_classification_logits, labels - 1)
            loss += classification_loss_value
            # writer.add_scalar('Loss/classification_loss', classification_loss_value.item(), epoch * len(train_loader) + step)

        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if rank == 0:
            pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, train total loss: {loss.item():.4f}")
            
    return np.mean(train_losses), train_losses


def eval_epoch_batch(eval_loader, model, loss_funs, loss_coefs, epoch = None):
    
    with torch.no_grad():
        eval_losses, seg_losses, ce_losses, clf_losses, dice_coeffs = [], [], [], [], []
        preds, trues = [], []
         # Get the rank of the current process
        rank = dist.get_rank()

        # Only create a progress bar if this is the main process
        if rank == 0:
            pbar = tqdm(eval_loader)
        else:
            pbar = eval_loader  # Just use the loader without tqdm
            
        for step, batch in enumerate(pbar):
            # one batch
            image, gt2D, tokens, labels = batch["image"].to(device), batch["gt2D"].to(device), batch["token"].to(device), batch["label_id"].to(device)
            # forward pass
            medsam_pred, text_classification_logits = model(image, tokens)
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
            if rank == 0:
                pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, eval total loss: {total_loss:.4f}")
        
    return np.mean(eval_losses), np.mean(seg_losses), np.mean(ce_losses), np.mean(clf_losses), np.mean(dice_coeffs), np.concatenate(preds, axis = 0), np.concatenate(trues, axis = 0)


def test_epoch_batch(eval_loader, model, loss_funs, loss_coefs, epoch = None):
    with torch.no_grad():
        eval_losses, seg_losses, ce_losses, clf_losses, dice_coeffs = [], [], [], [], []
        preds, trues = [], []

        # Only create a progress bar if this is the main process
        pbar = tqdm(eval_loader)
            
        for step, batch in enumerate(pbar):
            # one batch
            image, gt2D, tokens = batch["image"].to(device), batch["gt2D"].to(device), batch["token"].to(device)#, batch["label_id"].to(device) 
            # forward pass
            medsam_pred, text_classification_logits = model(image, tokens)
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