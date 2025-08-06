# Script 2: Get images ready for semantic segmentation
# Base path: resolve relative to current script location
import os
import shutil
import random
from pathlib import Path


import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
base_path = Path.cwd()

'''
I will be preprocessing my images for semantic segmentation using the following steps:
- Changing the label of segmentation masks to [0,1,2,3] original was labels are 0, 1, 2, 4
- Reducing the number of images sliced we are presented with.
- Converting Each Person folder to contain a numpy format of tumour and corresponding masks. This involes mergeing channels, cropping, standardising and saving.
- Keeping the names of each numpy images associated with the patient # WILL DO IF I HAVE TIME
- T1 does not add much details for tumour training so gotten rid of it

'''

# PART 1: Load the correct paths and create a list of all images ####################################################################

TRAIN_DATASET_PATH = 'data/Train_data'
VALIDATION_DATASET_PATH = 'data/Validate_data'

t2_list = sorted(glob.glob(TRAIN_DATASET_PATH + '/*/*T2.nii*'))
t1GD_list = sorted(glob.glob(TRAIN_DATASET_PATH +'/*/*T1GD.nii*'))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH +'/*/*FLAIR.nii*'))
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH + '/*/*segm.nii*'))

print(len(t2_list), len(t1GD_list) ,len(flair_list), len(mask_list)) # checking these are all same length

# PART 2: For training images: Prepare images ###################################################################################

for img in range(len(t2_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)

    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

    temp_image_t1GD=nib.load(t1GD_list[img]).get_fdata()
    temp_image_t1GD=scaler.fit_transform(temp_image_t1GD.reshape(-1, temp_image_t1GD.shape[-1])).reshape(temp_image_t1GD.shape)

    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3



    temp_combined_images = np.stack([temp_image_flair, temp_image_t1GD, temp_image_t2], axis=3)

    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. # !!! NO LONGER APPLICABLE
    #cropping x, y, and z
    temp_combined_images = temp_combined_images[56:184, 56:184, 40:104]  # 104 - 40 = 64 so size (128, 128, 64)
    temp_mask = temp_mask[56:184, 56:184, 40:104]

    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume

        # One-hot encode the mask for 4
        num_classes = 4
        temp_mask = np.eye(num_classes)[temp_mask]

        image_path = Path(TRAIN_DATASET_PATH, 'Uimages')
        mask_path = Path(TRAIN_DATASET_PATH, 'Umasks')

        image_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        # Save files
        np.save(image_path / f'image_{img}.npy', temp_combined_images)
        np.save(mask_path / f'mask_{img}.npy', temp_mask)

# PART 3: For validation images: Prepare images and masks in loop ####################################################################################################
 # Note: This part is similar to the training part, but uses a different dataset path

vt2_list = sorted(glob.glob(VALIDATION_DATASET_PATH + '/*/*T2.nii*'))
vt1GD_list = sorted(glob.glob(VALIDATION_DATASET_PATH + '/*/*T1GD.nii*'))
vflair_list = sorted(glob.glob(VALIDATION_DATASET_PATH + '/*/*FLAIR.nii*'))
vmask_list = sorted(glob.glob(VALIDATION_DATASET_PATH + '/*/*segm.nii*'))

print(len(vt2_list), len(vt1GD_list) ,len(vflair_list), len(vmask_list)) # checking these are all same length

#Each volume generates 18 64x64x64x4 sub-volumes.
#Total 369 volumes = 6642 sub volumes


for img in range(len(vt2_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)

    temp_image_t2=nib.load(vt2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

    temp_image_t1GD=nib.load(vt1GD_list[img]).get_fdata()
    temp_image_t1GD=scaler.fit_transform(temp_image_t1GD.reshape(-1, temp_image_t1GD.shape[-1])).reshape(temp_image_t1GD.shape)

    temp_image_flair=nib.load(vflair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

    temp_mask=nib.load(vmask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3



    temp_combined_images = np.stack([temp_image_flair, temp_image_t1GD, temp_image_t2], axis=3)

    #Crop to a size to be divisible by 64 so we can later extract ____ patches.
    #cropping x, y, and z

    temp_combined_images = temp_combined_images[56:184, 56:184, 40:104]  # 104 - 40 = 64 so size (128, 128, 64)
    temp_mask = temp_mask[56:184, 56:184, 40:104]

    val, counts = np.unique(temp_mask, return_counts=True)
    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume

        # One-hot encode the mask for 4
        num_classes = 4
        temp_mask = np.eye(num_classes)[temp_mask]

        image_path = Path(VALIDATION_DATASET_PATH, 'Uimages')
        mask_path = Path(VALIDATION_DATASET_PATH, 'Umasks')

        image_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)

        # Save files
        np.save(image_path / f'image_{img}.npy', temp_combined_images)
        np.save(mask_path / f'mask_{img}.npy', temp_mask)
print("All images and masks have been prepared and saved successfully.")


