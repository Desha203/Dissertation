# preprocessing

# Change to correct directory

#Script 1: To organise folders

import os
import shutil
import random
from pathlib import Path

# Making directories
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# --- CONFIGURATION ---
# source_dir = Path("/content/drive/MyDrive/Dissertation/Segmentation_Masks")  # Corrected path
# subject_dir = Path("/content/drive/MyDrive/Dissertation/Mri_scans_images")   # Corrected path
# output_base = Path("data")

source_dir = Path("/user/home/ms13525/scratch/hackathon-2025/data/images_segm")  # Corrected path
subject_dir = Path("/user/home/ms13525/scratch/hackathon-2025/data/images_structural")   # Corrected path
output_base = Path("data")

pattern = "UPENN-GBM-*_segm.nii.gz"
train_ratio, val_ratio = 0.8, 0.1  # test will be the rest

# --- STEP 1: Find sample files ---
all_files = sorted(source_dir.glob(pattern))

print(len(all_files), "files found matching the pattern.")

selected_files = all_files

# --- STEP 2: Clean output directory ---
for split in ['Train_data', 'Validate_data', 'Test_data']:
    split_path = output_base / split
    if split_path.exists():
        shutil.rmtree(split_path)
    split_path.mkdir(parents=True, exist_ok=True)

# --- STEP 3: Split and copy ---
random.shuffle(selected_files)
total_files = len(selected_files)

# Correctly calculate the number of files for each split
num_train = int(train_ratio * total_files)
num_val = int(val_ratio * total_files)
num_test = total_files - num_train - num_val


# Ensure at least 1 file in each if total_files allows
if num_val == 0 and total_files >= 3:
    num_val = 1
    num_train = max(1, num_train - 1)

if num_train + num_val > total_files:
    num_test = 0 # will remove at some point

splits = {
    'Train_data': selected_files[:num_train],
    'Validate_data': selected_files[num_train:num_train + num_val],
    'Test_data': selected_files[num_train + num_val:]
}

for split_name, files in splits.items():
    for file in files:
        filename = file.name

        # Extract ID
        parts = filename.replace("UPENN-GBM-", "").split("_")
        subject_id = parts[0]
        subject_folder_name = f"UPENN-GBM-{subject_id}_11"  # have to do because some folders are repeated and end in _21
        subject_folder = subject_dir / subject_folder_name

        if not subject_folder.exists():
            print(f" Subject folder not found: {subject_folder}")
            continue

        # Copy subject folder
        dest_subject_folder = output_base / split_name / subject_folder_name
        shutil.copytree(subject_folder, dest_subject_folder, dirs_exist_ok=True)

        # Copy .nii.gz file into split root for now (will move it later)
        dest_nifti_temp = output_base / split_name / filename
        shutil.copy2(file, dest_nifti_temp)

print(" Step 3: Data split and folders copied.")


# --- STEP 4: Move .nii.gz files into their corresponding subject folders ---
print("Step 4: Moving .nii.gz files into their subject folders...")

for split in ['Train_data', 'Validate_data', 'Test_data']:
    split_path = output_base / split

    for file in split_path.glob("UPENN-GBM-*_*_segm.nii.gz"):
        filename = file.name

        # Extract subject ID
        parts = filename.replace("UPENN-GBM-", "").split("_")
        subject_id = parts[0]
        subject_folder_name = f"UPENN-GBM-{subject_id}_11"
        subject_folder = split_path / subject_folder_name

        if subject_folder.exists():
            dest_path = subject_folder / filename
            shutil.move(str(file), str(dest_path))
            print(f" Moved {filename} to {subject_folder}")
        else:
            print(f"Folder not found for {filename}: expected {subject_folder}")

print(" Done: All files organised and split.")

'''
Take Patient UPEN 0002 this is the type of format we are working with:
- Each Patient Has about 160 slices of their brain, each slice is in 4 different colour filters and an associated mask. Not all is needed. >> reduce slices from 50-100
- Most of the images are black spaces > crop to from 240,240,x to 128,128,64
- Images are not standardised
- Mask Labels are not labelled correctly
- T1 is not adding that much detail to tumour that T1GD is not already doing
'''
TRAIN_DATASET_PATH = 'data/Train_data'
VALIDATION_DATASET_PATH = 'data/Validate_data'

# test_image_flair=nib.load(TRAIN_DATASET_PATH +'/UPENN-GBM-00002_11/UPENN-GBM-00002_11_FLAIR.nii.gz').get_fdata()
# test_image_t1=nib.load(TRAIN_DATASET_PATH + '/UPENN-GBM-00002_11/UPENN-GBM-00002_11_T1.nii.gz').get_fdata()
# test_image_t1GD=nib.load(TRAIN_DATASET_PATH + '/UPENN-GBM-00002_11/UPENN-GBM-00002_11_T1GD.nii.gz').get_fdata()
# test_image_t2=nib.load(TRAIN_DATASET_PATH + '/UPENN-GBM-00002_11/UPENN-GBM-00002_11_T2.nii.gz').get_fdata()

# test_mask=nib.load(TRAIN_DATASET_PATH + '/UPENN-GBM-00002_11/UPENN-GBM-00002_11_segm.nii.gz').get_fdata()
# print(test_image_flair.max())
# print(np.unique(test_mask))









