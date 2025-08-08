'''
The purpose of this script is reorganise data from a source directory into training, validation, and test sets.
I will be performing the following steps:
-importing necessary libraries.
-creating directories data (containing training, validation, and test directories) and results.
-setting seed for reproducibility.
-randomly selecting patient folders from source directory into training, validation, and test sets (80%, 10%, 10%).
-copying the subject folders from a source directory into the allocated directories.
-Ensuring each patient folders contained their images and corresponding masks in nifti format (.nii.gz).
-printing the number of files in each split.
'''

#importing necessary libraries
import os
import shutil
import random
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf

#setting random seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Making directories and setting paths  
os.makedirs("../data", exist_ok=True)
os.makedirs("../results", exist_ok=True)

source_dir = Path("/user/home/ms13525/scratch/hackathon-2025/data/images_segm")  # Directory with the NIfTI files
subject_dir = Path("/user/home/ms13525/scratch/hackathon-2025/data/images_structural")  # Directory with folders for each subject
output_base = Path("../data")


# training, validation, and test ratios
pattern = "UPENN-GBM-*_segm.nii.gz"
train_ratio, val_ratio = 0.8, 0.1  # test will be 0.1


#Finding all files matching the pattern
all_files = sorted(source_dir.glob(pattern))

print(len(all_files), "files found matching the pattern.")

selected_files = all_files

# splitting the files into training, validation, and test sets 
for split in ['Train_data', 'Validate_data', 'Test_data']:
    split_path = output_base / split
    if split_path.exists():
        shutil.rmtree(split_path)
    split_path.mkdir(parents=True, exist_ok=True)


random.shuffle(selected_files)
total_files = len(selected_files)

# Correctly calculate the number of files for each split
num_train = int(train_ratio * total_files)
num_val = int(val_ratio * total_files)
num_test = total_files - num_train - num_val


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
        subject_folder_name = f"UPENN-GBM-{subject_id}_11"  # have to do because some folders are repeated and end in _21 # REMOVE THIS COMMENT AS WE ARE USING MANUALLY SEGMENTED TRUTHS
        subject_folder = subject_dir / subject_folder_name

        if not subject_folder.exists():
            print(f"Subject folder not found: {subject_folder}")
            continue

        # Copy subject folder
        dest_subject_folder = output_base / split_name / subject_folder_name
        shutil.copytree(subject_folder, dest_subject_folder, dirs_exist_ok=True)

        # Copy .nii.gz file into split root for now (will move it later)
        dest_nifti_temp = output_base / split_name / filename
        shutil.copy2(file, dest_nifti_temp)

print(" Step 3: Data split and folders copied.")

# Transfer files into their subject folders

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
            print(f" Moved {filename} → {subject_folder}")
        else:
            print(f"Folder not found for {filename}: expected {subject_folder}")

print(" Done: All files organised and split.")
print(len(os.listdir(output_base / 'Train_data')), "training files")
print(len(os.listdir(output_base / 'Validate_data')), "validation files")
print(len(os.listdir(output_base / 'Test_data')), "test files")
