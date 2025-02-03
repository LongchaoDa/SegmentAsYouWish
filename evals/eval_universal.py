import json
from scipy.ndimage import distance_transform_edt, sobel
import nibabel as nib
import numpy as np
import numpy as np
import os

with open("prompts/free_form_text_prompts.json", 'r') as file:
    free_form_text_prompts = json.load(file)
    
flare22_labelids = {list(free_form_text_prompts['FLARE22_train'][key].keys())[0]:key for key in free_form_text_prompts['FLARE22_train']}
word_labelids = {list(free_form_text_prompts['WORD_test'][key].keys())[0]:key for key in free_form_text_prompts['WORD_test']}
raos_labelids = {list(free_form_text_prompts['RAOS'][key].keys())[0]:key for key in free_form_text_prompts['RAOS']}

def evaluate_slices(gt_data, pred_data, label_value=2):
    """
    Evaluate Dice and NSD slice-by-slice for liver segmentation.
    
    Args:
    gt_data: Ground truth 3D volume with multiple labels.
    pred_data: Predicted 3D volume with binary mask (0 for background, 1 for liver).
    label_value: The label value of liver in the ground truth.
    tolerance: Distance tolerance for NSD calculation (in voxels).
    
    Returns:
    Dice and NSD scores for each slice.
    """
    # Get the number of slices in the z-dimension (axial slices)
    num_slices = gt_data.shape[2]
    
    dice_scores = []
    nsd_scores = []
    
    for i in range(num_slices):
        gt_slice = gt_data[:, :, i]  # Extract a 2D slice from the ground truth
        pred_slice = pred_data[:, :, i]  # Extract a 2D slice from the prediction
        
        # Calculate Dice and NSD for this slice
        dice_scores.append(dice_coefficient(gt_slice, pred_slice,  label_value))
        nsd_scores.append(normalized_surface_distance_np(gt_slice, pred_slice, label_value, logits = False))
    
    return dice_scores, nsd_scores

def load_nifti_file(file_path):
    """
    Load a NIfTI file and return the image data as a numpy array.
    """
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data

def dice_coefficient(preds, targets,  label_value, smooth=0.1):
    # Flatten the tensors
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    targets = (targets == label_value).astype(np.uint8)

    # Calculate Dice Coefficient
    # print(preds.shape, targets.shape)
    intersection = (preds * targets).sum()
    dice_coeff = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    return dice_coeff.item()

def extract_boundary_np(segmentation):
    """
    Extract the boundary (surface) pixels from the segmentation map using Sobel filter.
    Args:
        segmentation (np.ndarray): Binary segmentation map of shape (H, W).
    Returns:
        np.ndarray: Binary map of boundary pixels of shape (H, W).
    """
    # Apply Sobel filter to extract edges
    grad_x = sobel(segmentation, axis=0)
    grad_y = sobel(segmentation, axis=1)
    
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    boundary = (gradient_magnitude > 0).astype(np.float32)
    
    return boundary

def compute_surface_distances_np(seg_pred, seg_gt):
    """
    Compute the surface distance between two binary segmentation maps.
    Args:
        seg_pred (np.ndarray): Predicted binary segmentation map of shape (H, W).
        seg_gt (np.ndarray): Ground truth binary segmentation map of shape (H, W).
    Returns:
        np.ndarray: Array of distances from each boundary pixel in `seg_pred` to `seg_gt`.
    """
    # Extract boundaries
    boundary_pred = extract_boundary_np(seg_pred)
    boundary_gt = extract_boundary_np(seg_gt)
    
    # Compute the distance transform of the boundaries
    dist_gt_to_pred = distance_transform_edt(1 - boundary_pred)
    dist_pred_to_gt = distance_transform_edt(1 - boundary_gt)
    
    # Find the distances for boundary points only
    distances_gt_to_pred = dist_pred_to_gt[boundary_pred.astype(bool)]
    distances_pred_to_gt = dist_gt_to_pred[boundary_gt.astype(bool)]
    
    return distances_gt_to_pred, distances_pred_to_gt

def normalized_surface_distance_np(seg_pred_logits, seg_gt, label_value, threshold=0.5, distance_threshold=1.0):
    """
    Calculate the Normalized Surface Distance (NSD) between the logits of predicted segmentation
    and the ground truth segmentation maps.
    Args:
        seg_pred_logits (np.ndarray): Predicted logits of shape (H, W).
        seg_gt (np.ndarray): Ground truth binary segmentation map of shape (H, W).
        threshold (float): Threshold value to convert logits to binary mask.
        distance_threshold (float): Distance threshold for computing NSD.
    Returns:
        float: NSD score.
    """
    seg_gt = (seg_gt == label_value).astype(np.uint8)
    
    seg_pred = seg_pred_logits
    
    # Compute surface distances between predicted and ground truth boundaries
    distances_gt_to_pred, distances_pred_to_gt = compute_surface_distances_np(seg_pred, seg_gt)
    
    # Normalize distances and compute the fraction below the distance threshold
    num_within_threshold_gt_to_pred = (distances_gt_to_pred < distance_threshold).sum()
    num_within_threshold_pred_to_gt = (distances_pred_to_gt < distance_threshold).sum()
    
    total_boundary_points = len(distances_gt_to_pred) + len(distances_pred_to_gt)
    
    nsd = (num_within_threshold_gt_to_pred + num_within_threshold_pred_to_gt) / (total_boundary_points + 1.0)
    
    return nsd


### Testing on FLARE22
flare_organ_names = [name.title() for name in list(flare22_labelids.keys())]
dice_scores, nsd_scores = [], []
for organ_name in flare_organ_names:
    for flare_name in os.listdir("/home/ec2-user/SageMaker/SegmentAsYouWish/text_prompt/evals/baseline_univeralmodel/"):
        if "FLARE22" in flare_name:
            gt_path = '/home/ec2-user/SageMaker/SegmentAsYouWish/text_prompt/data/touse/FLARE22/npy/CT_Abd_test/'+flare_name[:-4] +'_gt.nii.gz'
            path_liver = "/home/ec2-user/SageMaker/SegmentAsYouWish/text_prompt/evals/baseline_univeralmodel/"+flare_name+"/"+flare_name+"_" + organ_name + ".nii.gz"
            # Load the ground truth and predicted liver masks
            gt_data = load_nifti_file(gt_path)
            pred_data = load_nifti_file(path_liver)
            # Perform slice-level evaluation
            dice, nsd = evaluate_slices(gt_data, pred_data, label_value=float(flare22_labelids[organ_name.lower()]))
            dice_scores.append(np.mean(dice))
            nsd_scores.append(np.mean(nsd))
print("FLARE22", np.mean(dice_scores), np.mean(nsd_scores))

### Testing on WORD
word_organ_names = [name.title() for name in list(word_labelids.keys())]
dice_scores, nsd_scores = [], []
for organ_name in word_organ_names:
    for file in os.listdir("/home/ec2-user/SageMaker/SegmentAsYouWish/text_prompt/evals/baseline_univeralmodel/"):
        if "CT_WORD" in file:
            gt_path = '/home/ec2-user/SageMaker/SegmentAsYouWish/text_prompt/data/touse/WORD/npy/WORD-V0.1.0/CT_WORD/Val/'+file[:-3]+'gt.nii.gz'
            pred_path = "/home/ec2-user/SageMaker/SegmentAsYouWish/text_prompt/evals/baseline_univeralmodel/" + file  + "/" + file+"_" + organ_name + ".nii.gz"
            # Load the ground truth and predicted liver masks
            gt_data = load_nifti_file(gt_path)
            pred_data = load_nifti_file(pred_path)
            # Perform slice-level evaluation
            dice, nsd = evaluate_slices(gt_data, pred_data, label_value=float(word_labelids[organ_name.lower()]))
            dice_scores.append(np.mean(dice))
            nsd_scores.append(np.mean(nsd))
print("WORD", np.mean(dice_scores), np.mean(nsd_scores))

### Testing on RAOS
raos_organ_names = [name.title() for name in list(word_labelids.keys())]
# File paths
dice_scores = []
nsd_scores = []
for organ_name in raos_organ_names:
    for file in os.listdir("/home/ec2-user/SageMaker/SegmentAsYouWish/text_prompt/evals/baseline_univeralmodel/"):
        if "CT_RAOS" in file:
            gt_path = '/home/ec2-user/SageMaker/SegmentAsYouWish/text_prompt/data/touse/RAOS/npy/RAOS-Real/CT_RAOS/CancerImages(Set1)/Ts/'+file[:-3]+'gt.nii.gz'
            pred_path = "/home/ec2-user/SageMaker/SegmentAsYouWish/text_prompt/evals/baseline_univeralmodel/" + file  + "/" + file + "_" + organ_name + ".nii.gz"
            # Load the ground truth and predicted liver masks
            gt_data = load_nifti_file(gt_path)
            pred_data = load_nifti_file(pred_path)
            # Perform slice-level evaluation
            dice, nsd = evaluate_slices(gt_data, pred_data, label_value=float(word_labelids[organ_name.lower()]))
            dice_scores.append(np.mean(dice))
            nsd_scores.append(np.mean(nsd))
print("RAOS", np.mean(dice_scores), np.mean(nsd_scores))