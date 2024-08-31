import os
import shutil

def organize_files(parent_path, root_path):
    # Define paths for saving images and masks
    images_path = os.path.join(root_path, "images")
    masks_path = os.path.join(root_path, "labels")
    
    # Create directories if they don't exist
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(masks_path, exist_ok=True)
    
    # Iterate through all files in the parent path
    for filename in os.listdir(parent_path):
        full_file_path = os.path.join(parent_path, filename)
        
        # Check if it's a file (not a directory)
        if os.path.isfile(full_file_path):
            # Check if it's an image file
            if filename.startswith("volume-") and filename.endswith(".nii"):
                shutil.move(full_file_path, os.path.join(images_path, filename))
            # Check if it's a mask file
            elif filename.startswith("segmentation-") and filename.endswith(".nii"):
                shutil.move(full_file_path, os.path.join(masks_path, filename))

# Example usage
parent_path = "data/2LiTS/media/nas/01_Datasets/CT/LITS/Training Batch 1"
root_path = "data/2LiTS"

organize_files(parent_path, root_path)


# labels: (512, 512, 123)
# images: (512, 512, 123)
