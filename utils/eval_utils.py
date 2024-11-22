import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import numpy as np
import torch.nn as nn
from scipy.ndimage import convolve, distance_transform_edt, sobel


def visualize_seg(img_tensor, seg_tensor, preds_tensor):
    # # Assuming your image tensor is `img_tensor`, segmentation tensor is `seg_tensor`, 
    # and prediction tensor (logits) is `preds_tensor`
    # img_tensor: shape (3, 1024, 1024), seg_tensor: shape (1, 256, 256), preds_tensor: shape (1, 256, 256)

    # Convert image tensor to numpy array and permute to (H, W, C) for visualization
    img_np = img_tensor.permute(1, 2, 0).numpy()

    # Normalize image if needed (ensure values are in [0, 1])
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # Resize the ground truth segmentation and predictions to match image size
    seg_resized = T.Resize((1024, 1024), interpolation=T.InterpolationMode.NEAREST)(seg_tensor)
    preds_resized = T.Resize((1024, 1024), interpolation=T.InterpolationMode.NEAREST)(preds_tensor)

    # Convert ground truth and predicted segmentation to numpy arrays
    seg_np = seg_resized.squeeze().numpy()
    preds_np = preds_resized.squeeze().numpy()

    # Apply threshold to the predicted logits (e.g., 0.5) to get binary predictions
    preds_binary = preds_np > 0.5

    # Plot the image with both ground truth and predicted segmentation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display the original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Overlay the ground truth segmentation mask on the image
    axes[1].imshow(img_np)
    axes[1].imshow(seg_np, cmap='Reds', alpha=0.25)  # Ground truth mask in red
    axes[1].set_title('Ground Truth Segmentation')
    axes[1].axis('off')

    # Overlay the predicted segmentation mask on the image
    axes[2].imshow(img_np)
    axes[2].imshow(preds_binary, cmap='Blues', alpha=0.25)  # Prediction mask in blue
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def apply_threshold(logits, threshold=0.5):
    """
    Apply sigmoid and threshold to convert logits to a binary segmentation map.
    Args:
        logits (np.ndarray): Logits output from the model of shape (H, W).
        threshold (float): Threshold value to convert logits to binary mask.
    Returns:
        np.ndarray: Binary segmentation map of shape (H, W).
    """
    probs = sigmoid(logits)  # Convert logits to probabilities
    binary_segmentation = (probs >= threshold).astype(np.float32)  # Apply threshold
    return binary_segmentation

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

def normalized_surface_distance_np(seg_pred_logits, seg_gt, threshold=0.5, distance_threshold=1.0, logits = True):
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
    if logits:
        # Convert logits to binary segmentation using the threshold
        seg_pred = apply_threshold(seg_pred_logits, threshold)
    else:
        seg_pred = seg_pred_logits
    
    # Compute surface distances between predicted and ground truth boundaries
    distances_gt_to_pred, distances_pred_to_gt = compute_surface_distances_np(seg_pred, seg_gt)
    
    # Normalize distances and compute the fraction below the distance threshold
    num_within_threshold_gt_to_pred = (distances_gt_to_pred < distance_threshold).sum()
    num_within_threshold_pred_to_gt = (distances_pred_to_gt < distance_threshold).sum()
    
    total_boundary_points = len(distances_gt_to_pred) + len(distances_pred_to_gt)
    
    nsd = (num_within_threshold_gt_to_pred + num_within_threshold_pred_to_gt) / total_boundary_points
    
    return nsd

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

# def compute_dice_coefficient(mask_gt, mask_pred):
#     """Compute soerensen-dice coefficient.

#     compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
#     and the predicted mask `mask_pred`. 

#     Args:
#     mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
#     mask_pred: 3-dim Numpy array of type bool. The predicted mask.

#     Returns:
#     the dice coeffcient as float. If both masks are empty, the result is NaN
#     """
#     volume_sum = mask_gt.sum() + mask_pred.sum()
#     if volume_sum == 0:
#         return np.NaN
#     volume_intersect = (mask_gt & mask_pred).sum()
#     return 2*volume_intersect / volume_sum


# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.0):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         # Apply sigmoid if necessary (for binary classification)
#         probs = torch.sigmoid(logits) 

#         # Flatten the tensors
#         probs = probs.view(-1)
#         targets = targets.view(-1)

#         # Calculate the Dice coefficient
#         intersection = (probs * targets).sum()
#         dice_coeff = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

#         # Dice loss is 1 - Dice coefficient
#         return 1-dice_coeff
    
    
