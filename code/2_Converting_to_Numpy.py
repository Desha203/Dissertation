'''
The purpose of this script is to convert medical imaging data from NIfTI format (.nii.gz) into format required for 3D segmentation.

I will be pre-processing my training and validation images for semantic segmentation using the following steps:
- Importing necessary libraries.
- Relabeling segmentation masks: Changing the original labels from [0, 1, 2, 4] to [0, 1, 2, 3].
- Converting mask data type: Ensuring all masks are of type np.uint8.
- each subject, converting their 5 MRI modalities and corresponding mask into NumPy format.
- Cropping all volumes to a size of (128, 128, 128).
- Standardising the image intensities.
- Only saving image volumes that contain at least 1% tumour tissue.

'''

#importing necessary libraries
import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# making directories and setting paths
TRAIN_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
VALIDATION_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'

# Preprocessing to Numpy format for needed for segmentation

# Get list of patient folders
train_patient_folders = sorted(os.listdir(TRAIN_DATASET_PATH))

for idx, folder_name in enumerate(train_patient_folders):
    folder_path = os.path.join(TRAIN_DATASET_PATH, folder_name)
    patient_id = folder_name

    print(f"Processing {patient_id} ({idx + 1}/{len(train_patient_folders)})")

    # File paths - Use glob to be more robust to variations in file naming and extension
    t1_path = glob.glob(os.path.join(folder_path, f"{patient_id}_T1.nii*"))
    t2_path = glob.glob(os.path.join(folder_path, f"{patient_id}_T2.nii*"))
    t1gd_path = glob.glob(os.path.join(folder_path, f"{patient_id}_T1GD.nii*"))
    flair_path = glob.glob(os.path.join(folder_path, f"{patient_id}_FLAIR.nii*"))
    mask_path = glob.glob(os.path.join(folder_path, f"{patient_id}_segm.nii*"))


    # Check if all files exist
    if not (t1_path and t2_path and t1gd_path and flair_path and mask_path):
        print(f"Skipping {patient_id} due to missing files.")
        continue

    # Take the first match from glob results
    t1_path = t1_path[0]
    t2_path = t2_path[0]
    t1gd_path = t1gd_path[0]
    flair_path = flair_path[0]
    mask_path = mask_path[0]


   # Load and preprocess
    temp_image_t1 = nib.load(t1_path).get_fdata()
    t1 = scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)

    temp_image_t2 = nib.load(t2_path).get_fdata()
    t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

    temp_image_t1GD = nib.load(t1gd_path).get_fdata() # Corrected variable name
    t1GD = scaler.fit_transform(temp_image_t1GD.reshape(-1, temp_image_t1GD.shape[-1])).reshape(temp_image_t1GD.shape) # Corrected variable name

    temp_image_flair = nib.load(flair_path).get_fdata()
    flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

    temp_mask = nib.load(mask_path).get_fdata()
    mask = temp_mask.astype(np.uint8)
    mask[mask == 4] = 3  # Reassign label 4 to 3

    # Crop to (128, 128, 128)
    crop = (slice(56, 184), slice(56, 184), slice(13, 141))
    mask = mask[crop]  # crop before one-hot encoding

    # One-hot encode after cropping
    num_classes = 4
    one_hot_mask = np.eye(num_classes)[mask]  # shape: (128, 128, 128, 4)

    # Crop image modalities
    t1 = t1[crop]
    t2 = t2[crop]
    t1GD = t1GD[crop]
    flair = flair[crop]


      # Check tumour volume: must be >1% non-background

    vals, counts = np.unique(mask, return_counts=True)
    total_voxels = counts.sum()

    # Get background count (label 0) if it exists
    if 0 in vals:
                background_count = counts[np.where(vals == 0)[0][0]]
    else:
         background_count = 0

    # Keep the mask only if non-background makes up more than 1%
    if (1 - (background_count / total_voxels)) > 0.01:

     np.save(os.path.join(folder_path, "t1.npy"), t1)
     np.save(os.path.join(folder_path, "t2.npy"), t2)
     np.save(os.path.join(folder_path, "t1GD.npy"), t1GD)
     np.save(os.path.join(folder_path, "flair.npy"), flair)
     np.save(os.path.join(folder_path, "mask.npy"), one_hot_mask)


    print(f"✔ Finished and replaced .nii.gz files for training {patient_id}")

# now the same for validation set

# Get list of validation patient folders
validate_patient_folders = sorted(os.listdir(VALIDATION_DATASET_PATH))

for idx, folder_name in enumerate(validate_patient_folders):
    folder_path = os.path.join(VALIDATION_DATASET_PATH, folder_name)
    patient_id = folder_name

    print(f"Processing {patient_id} ({idx + 1}/{len(validate_patient_folders)})")

    # File paths - Use glob to be more robust to variations in file naming and extension
    t1_path = glob.glob(os.path.join(folder_path, f"{patient_id}_T1.nii*"))
    t2_path = glob.glob(os.path.join(folder_path, f"{patient_id}_T2.nii*"))
    t1gd_path = glob.glob(os.path.join(folder_path, f"{patient_id}_T1GD.nii*"))
    flair_path = glob.glob(os.path.join(folder_path, f"{patient_id}_FLAIR.nii*"))
    mask_path = glob.glob(os.path.join(folder_path, f"{patient_id}_segm.nii*"))


    # Check if all files exist
    if not (t1_path and t2_path and t1gd_path and flair_path and mask_path):
        print(f"Skipping {patient_id} due to missing files.")
        continue

    # Take the first match from glob results
    t1_path = t1_path[0]
    t2_path = t2_path[0]
    t1gd_path = t1gd_path[0]
    flair_path = flair_path[0]
    mask_path = mask_path[0]


   # Load and preprocess
    temp_image_t1 = nib.load(t1_path).get_fdata()
    t1 = scaler.fit_transform(temp_image_t1.reshape(-1, temp_image_t1.shape[-1])).reshape(temp_image_t1.shape)

    temp_image_t2 = nib.load(t2_path).get_fdata()
    t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

    temp_image_t1GD = nib.load(t1gd_path).get_fdata() # Corrected variable name
    t1GD = scaler.fit_transform(temp_image_t1GD.reshape(-1, temp_image_t1GD.shape[-1])).reshape(temp_image_t1GD.shape) # Corrected variable name

    temp_image_flair = nib.load(flair_path).get_fdata()
    flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

    temp_mask=nib.load(mask_path).get_fdata()
    mask=temp_mask.astype(np.uint8)
    mask[mask==4] = 3  #Reassign mask values 4 to 3

    # Crop to (128, 128, 128)
    crop = (slice(56, 184), slice(56, 184), slice(13, 141))
    mask = mask[crop]  # crop before one-hot encoding

    # One-hot encode after cropping
    num_classes = 4
    one_hot_mask = np.eye(num_classes)[mask]  # shape: (128, 128, 128, 4)

    # Crop to (128, 128, 128)

    t1 = t1[crop]
    t2 = t2[crop]
    t1GD = t1GD[crop] # Corrected variable name
    flair = flair[crop]

    # Check tumour volume: must be >1% non-background


    vals, counts = np.unique(mask, return_counts=True)
    total_voxels = counts.sum() # Define total_voxels here
    if 0 in vals:
                 background_count = counts[np.where(vals == 0)[0][0]]
    else:
         background_count = 0


    if (1 - (background_count / total_voxels)) > 0.01:
      #mask_cat = to_categorical(mask, num_classes=4).astype(np.uint8)
      np.save(os.path.join(folder_path, "t1.npy"), t1)
      np.save(os.path.join(folder_path, "t2.npy"), t2)
      np.save(os.path.join(folder_path, "t1GD.npy"), t1GD)
      np.save(os.path.join(folder_path, "flair.npy"), flair)
      np.save(os.path.join(folder_path, "mask.npy"), one_hot_mask)



    print(f"✔ Finished and replaced .nii.gz files for validation {patient_id}")
