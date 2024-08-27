import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom import dcmread
import SimpleITK as sitk
import shutil
from skimage.transform import resize

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom import dcmread
from skimage.transform import resize
import os

def read_dicom_folder(folder_path):
    # Get all DICOM files in the folder
    dicom_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.dcm')]
    dicom_files.sort()  # Ensure files are in the correct order
    
    # Read all DICOM slices and stack them
    slices = [dcmread(dcm_file) for dcm_file in dicom_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Sort by slice location
    image_data = np.stack([s.pixel_array for s in slices])
    
    return image_data

def adjust_image_slices(image_data, target_slices):
    current_slices = image_data.shape[0]
    if current_slices > target_slices:
        start_slice = (current_slices - target_slices) // 2
        adjusted_image_data = image_data[start_slice:start_slice + target_slices]
    elif current_slices < target_slices:
        pad_before = (target_slices - current_slices) // 2
        pad_after = target_slices - current_slices - pad_before
        adjusted_image_data = np.pad(image_data, ((pad_before, pad_after), (0, 0), (0, 0)), 'constant', constant_values=0)
    else:
        adjusted_image_data = image_data
    return adjusted_image_data

def show_and_save_dicom_nii_overlay(dicom_folder_path, label_nii_path, output_dir, slice_id=None, lower_bound=-240.0, upper_bound=160.0):
    # Read the DICOM folder and NIfTI file
    dicom_data = read_dicom_folder(dicom_folder_path)


    save_tag = label_nii_path.split("/")[-1].replace(".nii.gz", "").replace("label", "")

    label_sitk = sitk.ReadImage(label_nii_path)
    
    # Get the label data
    label_data = sitk.GetArrayFromImage(label_sitk)
    
    # Adjust the DICOM data to match the number of slices in the label
    dicom_data_aligned = adjust_image_slices(dicom_data, label_data.shape[0])

    # Adjust window width and level for the DICOM image
    dicom_data_pre = np.clip(dicom_data_aligned, lower_bound, upper_bound)
    dicom_data_pre = (dicom_data_pre - np.min(dicom_data_pre)) / (np.max(dicom_data_pre) - np.min(dicom_data_pre)) * 255.0
    dicom_data_pre = np.uint8(dicom_data_pre)
    
    # Select the slice
    if slice_id is None:
        slice_id = int(dicom_data_pre.shape[0] / 2)
    
    dicom_slice = dicom_data_pre[slice_id]
    label_slice = label_data[slice_id]
    
    # Resize DICOM image to match label dimensions (if necessary)
    if dicom_slice.shape != label_slice.shape:
        dicom_slice_resized = resize(dicom_slice, label_slice.shape, anti_aliasing=True)
    else:
        dicom_slice_resized = dicom_slice
    
    # Overlay the label on the DICOM slice
    overlay = dicom_slice_resized.copy()
    overlay[label_slice > 0] = 255  # Set the label area to maximum intensity (white)
    
    # Show the DICOM and overlay slices side by side
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].imshow(np.rot90(dicom_slice_resized, 2), cmap='gray')
    # axes[0].set_title(f'DICOM Slice: {slice_id}')
    # axes[0].axis('off')
    
    # axes[1].imshow(np.rot90(dicom_slice_resized, 2), cmap='gray')
    # axes[1].imshow(np.rot90(label_slice, 2), cmap='viridis', alpha=0.5)  # Overlay with transparency
    # axes[1].set_title(f'Overlay Slice: {slice_id}')
    # axes[1].axis('off')
    
    # plt.show()

    # Save the processed DICOM slice and label as npy files
    os.makedirs(os.path.join(output_dir, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "gts"), exist_ok=True)
    
    img_output_path = os.path.join(output_dir, "imgs", f"image_{save_tag}.npy")
    label_output_path = os.path.join(output_dir, "gts", f"label_{save_tag}.npy")
    
    np.save(img_output_path, dicom_slice_resized)
    np.save(label_output_path, label_slice)

    print(f"Saved DICOM slice as {img_output_path} and label as {label_output_path}")

def process_all_images_and_labels(root_folder, label_path, work_dir):
    for label_file in sorted(os.listdir(label_path)):
        if label_file.startswith("label") and label_file.endswith(".nii.gz"):
            label_number = label_file.split("label")[1].split(".nii.gz")[0]
            image_folder = os.path.join(root_folder, f"PANCREAS_{label_number}")

            if os.path.isdir(image_folder):
                label_nii_path = os.path.join(label_path, label_file)

                for subdir, dirs, files in os.walk(image_folder):
                    if any(f.endswith('.dcm') for f in files):
                        dicom_folder_path = subdir

                        show_and_save_dicom_nii_overlay(
                            dicom_folder_path=dicom_folder_path, 
                            label_nii_path=label_nii_path,
                            output_dir=work_dir
                        )

                        print(f"Processed and saved {dicom_folder_path} with {label_nii_path}")

# Usage
root_folder = "/home/local/ASURITE/longchao/Desktop/project/GE_health/SegmentAsYouWish/data/1Pancreas/download/Pancreas-CT-20200910/Pancreas-CT"
label_path = "/home/local/ASURITE/longchao/Desktop/project/GE_health/SegmentAsYouWish/data/1Pancreas/download/TCIA_pancreas_labels-02-05-2017"
work_dir = "/home/local/ASURITE/longchao/Desktop/project/GE_health/SegmentAsYouWish/data/1Pancreas/npy/"

process_all_images_and_labels(root_folder, label_path, work_dir)
