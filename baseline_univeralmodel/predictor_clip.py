import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader_without_gt, get_loader_without_gt_npy
from utils import loss
from utils.utils import dice_score, threshold_organ, visualize_label, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process, threshold_organ, save_results
import clip 
import sys 


device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append("/home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model")

torch.multiprocessing.set_sharing_strategy('file_system')

# update your organ list, please check in the Name_DIC:
# organ_list_to_seg = [6] # the organs you want to segment, input is the label


organ_list_to_seg = ["I want to see the Liver"]  # the organs you want to segment, input is the label
seg_text_inputs = torch.cat([clip.tokenize(item) for item in organ_list_to_seg]).to(device)


def get_encoded_result(Name_DIC, device):
    """
    This is following the paper's way to encode the prompt meaning in to the CLIP fearure space 

    return a list of features 
    """

    encoded_result = {}

    model_clip, preprocess = clip.load('ViT-B/32', device=device)

    sorted_items = sorted(Name_DIC.items(), key=lambda item: item[1])

    text_prompts = [f'A computerized tomography of a {organ}' for organ, _ in sorted_items]

    text_tokens = clip.tokenize(text_prompts).to(device)

    # 编码文本提示以获取嵌入向量
    with torch.no_grad():
        text_embeddings = model_clip.encode_text(text_tokens).float()  # 确保为 float32

    # normalize 
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    for (_, organ_id), embedding in zip(sorted_items, text_embeddings):
        encoded_result[organ_id] = embedding.cpu()  # 移动到 CPU 以节省 GPU 内存

    return encoded_result


def find_closest_id(encoded_result, new_texts, model_clip, device="cuda"):
    """
    this is takig a text as input (your query text)
    and then match the cloest one in the embedding that given by the above function 

    then extract the correspdonig id and return it
    
    """

    text_tokens = clip.tokenize(new_texts).to(device)

    with torch.no_grad():
        new_embeddings = model_clip.encode_text(text_tokens).float()

    # same embedding 
    new_embeddings /= new_embeddings.norm(dim=-1, keepdim=True)

    # 将所有 encoded_result 的嵌入堆叠
    organ_ids = list(encoded_result.keys())
    organ_embeddings = torch.stack([encoded_result[organ_id] for organ_id in organ_ids]).to(device)

    # cos simm
    similarity = new_embeddings @ organ_embeddings.T  # size: [batch_size, num_organs]

    #
    best_indices = similarity.argmax(dim=1).cpu()
    closest_ids = [organ_ids[idx.item()] for idx in best_indices]

    return closest_ids



Name_DIC = {
    'Spleen': 1,
    'Right Kidney': 2,
    'Left Kidney': 3,
    'Gall Bladder':4,
    'Esophagus': 5, 
    'Liver': 6,
    'Stomach': 7,
    'Arota': 8, 
    'Postcava': 9, 
    'Portal Vein and Splenic Vein': 10,
    'Pancreas': 11, 
    'Right Adrenal Gland': 12, 
    'Left Adrenal Gland': 13, 
    'Duodenum': 14, 
    'Hepatic Vessel': 15,
    'Right Lung': 16, 
    'Left Lung': 17, 
    'Colon': 18, 
    'Intestine': 19, 
    'Rectum': 20, 
    'Bladder': 21, 
    'Prostate': 22, 
    'Left Head of Femur': 23, 
    'Right Head of Femur': 24, 
    'Celiac Truck': 25,
    'Kidney Tumor': 26, 
    'Liver Tumor': 27, 
    'Pancreas Tumor': 28, 
    'Hepatic Vessel Tumor': 29, 
    'Lung Tumor': 30, 
    'Colon Tumor': 31, 
    'Kidney Cyst': 32
}


# Here we extract the best id and assign back to the id list: 

encoded_result = get_encoded_result(Name_DIC, device=device)
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
organ_list_to_seg = ["I want to see the Liver"]  # List of texts

# close id:
closest_ids = find_closest_id(encoded_result, organ_list_to_seg, model_clip, device=device)

# set back to the organ_seg id:
organ_list_to_seg = closest_ids


def visualize_predictions(batch, organ_list):
    """
    Visualize the predicted results stored in batch['results'].
    
    Args:
    - batch: The batch dictionary output containing the predictions under 'results'.
    - organ_list: The list of organs (classes) to visualize.
    
    """
    results = batch['results'].cpu().numpy()  # Convert to NumPy if it is still a tensor
    num_organs = results.shape[1]  # Assuming results are in [batch, organs, height, width, depth]

    # Visualize each organ's prediction
    for organ in organ_list:
        organ_index = organ - 1  # Organ indexes are 1-based in your list, adjust to 0-based
        organ_results = results[:, organ_index]  # Get the predictions for the current organ
        
        # organ_results is of shape [batch, height, width, depth]
        # Loop through both depth slices (2 slices in your case)
        for slice_idx in range(organ_results.shape[-1]):  # Loop over the last dimension (depth)
            slice_image = organ_results[0, :, :, slice_idx]  # Assuming batch size is 1

            # Plot the slice
            plt.figure(figsize=(6, 6))
            plt.imshow(slice_image, cmap='gray')
            plt.title(f"Prediction for {ORGAN_NAME[organ_index]} (Slice {slice_idx})")
            plt.axis('off')
            plt.savefig("/home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/data_temp/save_pred_img/saved_" + str(slice_idx) + ".png", dpi=400)
            # plt.show()

# slice 38

def validation(model, ValLoader, val_transforms, args, organ_list):
    if not os.path.exists(args.result_save_path):
        os.makedirs(args.result_save_path)
    model.eval()
    for index, batch in enumerate(tqdm(ValLoader)):
        # print("index, batch:::::::")
        # print(index, batch)
        image, name = batch["image"].cuda(), batch["name"]
        # print(f"image shape is: {str(image.shape)}")
        # print("image, name ============")
        # print(image.shape, name)
        # exit("ahjiishcishiocshiovahsdvhoiasdhv")
        # print("----------------------------------------------------")
        # print(f"Image shape: {image.shape}, ROI: {(args.roi_x, args.roi_y, args.roi_z)}")
        # exit("ahjiishcishiocshiovahsdvhoiasdhv")
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)

        # print("pred_sigmoid::::::::=================")
        # print(pred_sigmoid)
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1],args)
        pred_hard_post = torch.tensor(pred_hard_post)

        batch['results'] = pred_hard_post

        print("=====almost finishing .......====")

        # visualize_predictions(batch, organ_list)

        # final_shape = batch['results'].shape 

        # print("final_shape//////..........")
        # print(final_shape)


        save_results(batch, args.result_save_path, val_transforms, organ_list)
            
        # torch.cuda.empty_cache()

def filter_folder_list(path):
    # List to store all folder names 
    folder_names = []

    # Iterate over all the files in the folder
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        
        # Check if it's a file (not a folder)
        if not os.path.isfile(file_path):
            folder_names.append(file)
    
    # Print the list of file names
    for file in folder_names:
        print(file)
    
    return folder_names


def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='Nvidia', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default='/home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/pretrained_weights/swinunetr.pth', help='The path resume from checkpoint')
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet]')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset

    # 
    parser.add_argument('--data_root_path', default="/home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/data_temp/simulate/npy/CT_Abd", help='data root path')
    parser.add_argument('--result_save_path', default="/home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/data_temp/simulate/npy/prediction", help='path for save result')



    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type= float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')

    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)

    args = parser.parse_args()

    # prepare the 3D model
    model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding='word_embedding'
                    )
    
    #Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['net']
    # args.epoch = checkpoint['epoch']
    num_count = 0
    for key, value in load_dict.items():
        if 'swinViT' in key or 'encoder' in key or 'decoder' in key:
            name = '.'.join(key.split('.')[1:])
            name = 'backbone.' + name
        else:
            name = '.'.join(key.split('.')[1:])
        store_dict[name] = value
        num_count += 1

    model.load_state_dict(store_dict)
    print('Use pretrained weights. load', num_count, 'params into', len(store_dict.keys()))

    model.cuda()

    torch.backends.cudnn.benchmark = True

    # test_loader, val_transforms = get_loader_without_gt_npy(args)
    test_loader, val_transforms = get_loader_without_gt(args)

    validation(model, test_loader, val_transforms, args, organ_list_to_seg)

    return args.result_save_path, args.data_root_path


def get_all_predicted_list(saved_folder_path):
    all_prediction_folders = filter_folder_list(saved_folder_path)
    organ_list_to_seg_names = []
    organ_list_to_seg

    for key, val in Name_DIC.items():
        print( key, val)
        for id in organ_list_to_seg:
            if id == val:
                organ_list_to_seg_names.append(key)

    print(organ_list_to_seg_names)

    file_name_list = []
    for f_name in all_prediction_folders:
        for organ_name in organ_list_to_seg_names:
            file_name = f_name + organ_name + ".nii.gz"
            full_path = os.path.join(saved_folder_path, f_name, file_name)
            file_name_list.append(full_path)

    # print("file_name_list.......")
    # print(file_name_list)
    # print(f"length of the list is: {len(file_name_list)}")
    return file_name_list

def get_all_gt_list(folder_path):
    gt_files = []

    # Iterate over all the files in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        # Check if it's a file (not a folder) and if it ends with '_gt.nii.gz'
        if os.path.isfile(file_path) and file.endswith("_gt.nii.gz"):
            gt_files.append(file)
    
    return gt_files

Dataset = {
    "FLARE22":
    {
        1: "liver",
        2: "right kidney",
        3: "spleen",
        4: "pancreas",
        5: "aorta",
        6: "inferior vena cava",
        7: "right adrenal gland",
        8: "left adrenal gland",
        9: "gallbladder",
        10: "esophagus",
        11: "stomach",
        12: "duodenum",
        13: "left kidney"
    }

}


if __name__ == "__main__":
    result_save_path, data_root_path = main()

    # get all of the file paths of predicted results
    file_name_list = get_all_predicted_list(result_save_path)
    gt_file_list = get_all_gt_list("/home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/data_temp/simulate/npy/CT_Abd")

    # organ_list_to_seg = [2, 3, 6]: in correspondance with [2, 3, 6] ==> Right Kidney, Left Kidney, Liver, so in FLARE22: 
    # Dataset_specific_labels = [2, 13, 1]







