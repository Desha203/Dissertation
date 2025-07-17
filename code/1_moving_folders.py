import os
import shutil
import random
from pathlib import Path

# Making directories 

os.makedirs("../data", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# --- CONFIGURATION ---
source_dir = Path("/user/home/ms13525/scratch/hackathon-2025/data/automated_segm")  # Directory with the NIfTI files
subject_dir = Path("/user/home/ms13525/scratch/hackathon-2025/data/images_structural")  # Directory with folders for each subject
output_base = Path("../data")  # Output root: will contain Train_data/, Validate_data/, Test_data/

pattern = "UPENN-GBM-*_automated_approx_segm.nii.gz"
num_samples = 250
train_ratio, val_ratio = 0.8, 0.1  # test will be the rest


# --- STEP 1: Find and randomly sample files ---
all_files = sorted(source_dir.glob(pattern)) # should find all 611 auto seg files

print(len(all_files), "files found matching the pattern.")

if len(all_files) < num_samples:
    raise ValueError(f"Found only {len(all_files)} matching files, need {num_samples}")

selected_files = random.sample(all_files, num_samples)


# --- STEP 2: Clean output directory ---
for split in ['Train_data', 'Validate_data', 'Test_data']:
    split_path = output_base / split
    if split_path.exists():
        shutil.rmtree(split_path)
    split_path.mkdir(parents=True, exist_ok=True)


# --- STEP 3: Split and copy ---
random.shuffle(selected_files)
num_train = int(train_ratio * num_samples)
num_val = int(val_ratio * num_samples)
splits = {
    'Train_data': selected_files[:num_train],
    'Validate_data': selected_files[num_train:num_train+num_val],
    'Test_data': selected_files[num_train+num_val:]
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
            print(f"⚠️ Subject folder not found: {subject_folder}")
            continue

        # Copy subject folder
        dest_subject_folder = output_base / split_name / subject_folder_name
        shutil.copytree(subject_folder, dest_subject_folder, dirs_exist_ok=True)

        # Copy .nii.gz file into split root for now (will move it later)
        dest_nifti_temp = output_base / split_name / filename
        shutil.copy2(file, dest_nifti_temp)

print("✅ Step 3: Data split and folders copied.")


# --- STEP 4: Move .nii.gz files into their corresponding subject folders ---
print("🔄 Step 4: Moving .nii.gz files into their subject folders...")

for split in ['Train_data', 'Validate_data', 'Test_data']:
    split_path = output_base / split

    for file in split_path.glob("UPENN-GBM-*_*_automated_approx_segm.nii.gz"):
        filename = file.name

        # Extract subject ID
        parts = filename.replace("UPENN-GBM-", "").split("_")
        subject_id = parts[0]
        subject_folder_name = f"UPENN-GBM-{subject_id}_11"
        subject_folder = split_path / subject_folder_name

        if subject_folder.exists():
            dest_path = subject_folder / filename
            shutil.move(str(file), str(dest_path))
            print(f"✅ Moved {filename} → {subject_folder}")
        else:
            print(f"⚠️ Folder not found for {filename}: expected {subject_folder}")

print("🎉 Done: All files organized and split.")
