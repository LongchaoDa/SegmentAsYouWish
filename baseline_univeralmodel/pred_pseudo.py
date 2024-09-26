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

torch.multiprocessing.set_sharing_strategy('file_system')


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
        print("index, batch:::::::")
        print(index, batch)
        image, name = batch["image"].cuda(), batch["name"]
        print(f"image shape is: {str(image.shape)}")
        print("image, name ============")
        print(image.shape, name)
        # exit("ahjiishcishiocshiovahsdvhoiasdhv")
        print("----------------------------------------------------")
        print(f"Image shape: {image.shape}, ROI: {(args.roi_x, args.roi_y, args.roi_z)}")
        # exit("ahjiishcishiocshiovahsdvhoiasdhv")
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian')
            pred_sigmoid = F.sigmoid(pred)

        print("pred_sigmoid::::::::=================")
        print(pred_sigmoid)
        pred_hard = threshold_organ(pred_sigmoid)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1],args)
        pred_hard_post = torch.tensor(pred_hard_post)

        batch['results'] = pred_hard_post

        print("=====almost finishing .......====")

        visualize_predictions(batch, organ_list)

        final_shape = batch['results'].shape 

        print("final_shape//////..........")
        print(final_shape)


        save_results(batch, args.result_save_path, val_transforms, organ_list)
            
        # torch.cuda.empty_cache()



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
    parser.add_argument('--resume', default='./pretrained_weights/swinunetr.pth', help='The path resume from checkpoint')
    parser.add_argument('--backbone', default='swinunetr', help='backbone [swinunetr or unet]')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--data_root_path', default=None, help='data root path')
    parser.add_argument('--result_save_path', default=None, help='path for save result')
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


    organ_list_to_seg = [6] # the organs you want to segment, input is the label

    validation(model, test_loader, val_transforms, args, organ_list_to_seg)

if __name__ == "__main__":
    main()

# /home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/data_temp

# python pred_pseudo.py --data_root_path /home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/data_temp/single_processed_npy --result_save_path /home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/data_temp/single_processed_npy/result
## For example: python pred_pseudo.py --data_root_path /home/data/ct/ --result_save_path /home/data/result



# 
# python pred_pseudo.py --data_root_path /home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/data_temp/niigz --result_save_path /home/local/ASURITE/longchao/Desktop/project/GE_health/CLIP-Driven-Universal-Model/data_temp/single_result 

# torch.Size([1, 1, 683, 683, 2]) ['CT_Abd_FLARE22_Tr_0001-050']
# torch.Size([1, 1, 683, 683, 2]) ['file_0001_050']

# torch.Size([1, 1, 218, 191, 183]) ['FLARE22_Tr_0001_0000']