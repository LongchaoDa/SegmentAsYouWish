import numpy as np
import random
import torch
import os
import kornia as K

# seed = 0
# os.environ['PYTHONHASHSEED']=str(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)


def apply_d4_transform(image, gts):
    """
    Apply a random D4 transformation to a NumPy array image in (C, H, W) format.
    D4 includes four rotations (0°, 90°, 180°, 270°) and four reflections.
    """
    choice = np.random.randint(0, 8)

    if choice == 0:
        # 0-degree rotation (identity)
        return image, gts
    elif choice == 1:
        # 90-degree rotation
        return np.rot90(image, k=1, axes=(1, 2)), np.rot90(gts, k=1, axes=(1, 2))
    elif choice == 2:
        # 180-degree rotation
        return np.rot90(image, k=2, axes=(1, 2)), np.rot90(gts, k=2, axes=(1, 2))
    elif choice == 3:
        # 270-degree rotation
        return np.rot90(image, k=3, axes=(1, 2)), np.rot90(gts, k=3, axes=(1, 2))
    elif choice == 4:
        # Horizontal flip
        return np.flip(image, axis=2), np.flip(gts, axis=2)
    elif choice == 5:
        # Vertical flip
        return np.flip(image, axis=1), np.flip(gts, axis=1)
    elif choice == 6:
        # Diagonal flip (transpose)
        return np.transpose(image, (0, 2, 1)), np.transpose(gts, (0, 2, 1))
    elif choice == 7:
        # Anti-diagonal flip (transpose + 90-degree rotation)
        return np.rot90(np.transpose(image, (0, 2, 1)), k=1, axes=(1, 2)), np.rot90(np.transpose(gts, (0, 2, 1)), k=1, axes=(1, 2))

    
def apply_d8_transform(image, gts):
    """
    Apply a random D8 transformation to a PyTorch tensor image in (C, H, W) format.
    D8 includes eight rotations (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    and eight reflections.
    """
    # Convert the image and gts from NumPy arrays to PyTorch tensors
    image_tensor = torch.from_numpy(image).unsqueeze(0).float()  # Ensure data type and add batch dimension
    gts_tensor = torch.from_numpy(gts).unsqueeze(0).float()      # Ensure data type and add batch dimension

    choice = np.random.randint(0, 16)

    if choice == 0:
        # 0-degree rotation (identity)
        rotated_image = image_tensor
        rotated_gts = gts_tensor
    elif choice in {1, 3, 5, 7}:
        # 45, 135, 225, 315 degree rotations
        degrees = (choice // 2) * 90 + 45
        rotated_image = K.geometry.transform.rotate(image_tensor, torch.tensor(degrees).float(), padding_mode='border')
        rotated_gts = K.geometry.transform.rotate(gts_tensor, torch.tensor(degrees).float(), padding_mode='border')
    elif choice in {2, 4, 6}:
        # 90, 180, 270 degree rotations
        degrees = ((choice - 2) // 2) * 90 + 90
        rotated_image = K.geometry.transform.rotate(image_tensor, torch.tensor(degrees).float(), padding_mode='border')
        rotated_gts = K.geometry.transform.rotate(gts_tensor, torch.tensor(degrees).float(), padding_mode='border')
    elif choice == 8 or choice == 9:
        # Horizontal or Vertical reflection
        flip_dim = -1 if choice == 8 else -2
        rotated_image = torch.flip(image_tensor, dims=[flip_dim])
        rotated_gts = torch.flip(gts_tensor, dims=[flip_dim])
    elif choice in {10, 11}:
        # Diagonal or Anti-diagonal reflection
        transposed_image = image_tensor.permute(0, 1, 3, 2)
        transposed_gts = gts_tensor.permute(0, 1, 3, 2)
        additional_rotation = 0 if choice == 10 else 90
        rotated_image = K.geometry.transform.rotate(transposed_image, torch.tensor(additional_rotation).float(), padding_mode='border')
        rotated_gts = K.geometry.transform.rotate(transposed_gts, torch.tensor(additional_rotation).float(), padding_mode='border')
    elif choice in {12, 13, 14, 15}:
        # Combined reflections and rotations
        flip_dim = -1 if choice in {12, 14} else -2
        additional_rotation = 45 if choice == 12 else 90 if choice == 13 else 135 if choice == 14 else 180
        flipped_image = torch.flip(image_tensor, dims=[flip_dim])
        flipped_gts = torch.flip(gts_tensor, dims=[flip_dim])
        rotated_image = K.geometry.transform.rotate(flipped_image, torch.tensor(additional_rotation).float(), padding_mode='border')
        rotated_gts = K.geometry.transform.rotate(flipped_gts, torch.tensor(additional_rotation).float(), padding_mode='border')

    # Convert tensors back to NumPy arrays
    return rotated_image.squeeze(0).cpu().numpy(), rotated_gts.squeeze(0).cpu().numpy()


