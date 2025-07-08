"""
Purpose of  this code is to get the dataset images ready for semantic segmentation. 
Code can be divided into a few parts....

#Combine 
#Changing mask pixel values (labels) from 4 to 3 (as the original labels are 0, 1, 2, 4)
#Visualize


https://pypi.org/project/nibabel/

All dataset multimodal scans are available as NIfTI files (.nii.gz) -> commonly used medical imaging format to store brain imagin data obtained using MRI and describe different MRI settings

#####################!UPDATE FOR MYSELF!###########################

T1: T1-weighted, native image, sagittal or axial 2D acquisitions, with 1–6 mm slice thickness.
T1c: T1-weighted, contrast-enhanced (Gadolinium) image, with 3D acquisition and 1 mm isotropic voxel size for most patients.
T2: T2-weighted image, axial 2D acquisition, with 2–6 mm slice thickness.
FLAIR: T2-weighted FLAIR image, axial, coronal, or sagittal 2D acquisitions, 2–6 mm slice thickness.

!UPDATE FOR MYSELF!

#Note: Segmented file name in Folder 355 has a weird name. Rename it to match others.
"""
from pathlib import Path
# Base path: resolve relative to current script location
base_path = Path(__file__).resolve().parent.parent / "data"

# Create the dataset directories if they don't exist
(base_path / "Train_data" / "UNET_Train_PATH").mkdir(parents=True, exist_ok=True)
(base_path / "Validate_data" / "UNET_Validate_PATH").mkdir(parents=True, exist_ok=True)

######################### START ###########################

# importing libraries 

import numpy as np
import nibabel as nib
import glob
#from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
#from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# PART 1: Load and preprocess using 1 example #################################################################### 

TRAIN_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
VALIDATION_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'
UNET_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/UNet_data/'

test_image_flair=nib.load(TRAIN_DATASET_PATH +'UPENN-GBM-00024_11/UPENN-GBM-00024_11_FLAIR.nii.gz').get_fdata()
print(test_image_flair.max()) # max pixel size was 629.0 so likely I will standardised to 1

#Scalers are applied to 1D so let us reshape and then reshape back to original shape. 
test_image_flair=scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)
print(test_image_flair.max()) # image scaled to 1 

# now same for other images T2,T1, T1GD but we will not standardise the mask 

test_image_t1=nib.load(TRAIN_DATASET_PATH + 'UPENN-GBM-00024_11/UPENN-GBM-00024_11_T1.nii.gz').get_fdata()
test_image_t1=scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

test_image_t1GD=nib.load(TRAIN_DATASET_PATH + 'UPENN-GBM-00024_11/UPENN-GBM-00024_11_T1GD.nii.gz').get_fdata()
test_image_t1GD=scaler.fit_transform(test_image_t1GD.reshape(-1, test_image_t1GD.shape[-1])).reshape(test_image_t1GD.shape)

test_image_t2=nib.load(TRAIN_DATASET_PATH + 'UPENN-GBM-00024_11/UPENN-GBM-00024_11_T2.nii.gz').get_fdata()
test_image_t2=scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

test_mask=nib.load(TRAIN_DATASET_PATH + 'UPENN-GBM-00024_11/UPENN-GBM-00024_11_automated_approx_segm.nii.gz').get_fdata()
test_mask=test_mask.astype(np.uint8)

print(np.unique(test_mask))  #0, 1, 2, 4 (We will reencode to 0, 1, 2, 3)
test_mask[test_mask==4] = 3  #Reassign mask values 4 to 3
print(np.unique(test_mask)) #done 

import random
n_slice=random.randint(0, test_mask.shape[2])

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(test_image_flair[:,:,n_slice], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(test_image_t1[:,:,n_slice], cmap='gray')
plt.title('Image t1')
plt.subplot(233)
plt.imshow(test_image_t1GD[:,:,n_slice], cmap='gray')
plt.title('Image t1GD')
plt.subplot(234)
plt.imshow(test_image_t2[:,:,n_slice], cmap='gray')
plt.title('Image t2')
plt.subplot(235)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')

plt.savefig("random_mri_slices.png")

##################################################
#PART 2: Explore the process of combining images to channels and divide them to patches
#Includes...
#Combining all 4 images to 4 channels of a numpy array.
#
################################################
#Flair, T1CE, annd T2 have the most information
#Combine t1ce, t2, and flair into single multichannel image

combined_x = np.stack([test_image_flair, test_image_t1GD, test_image_t2], axis=3)

print("Shape of combined_x:", combined_x.shape)
print("Size in MB:", combined_x.nbytes / (1024**2))

#Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
#cropping x, y, and z
#combined_x=combined_x[24:216, 24:216, 13:141]

combined_x=combined_x[56:184, 56:184, 13:141] #Crop to 128x128x128x4

print("Shape of combined_x:", combined_x.shape)
print("Size in MB:", combined_x.nbytes / (1024**2))

#Do the same for mask
test_mask = test_mask[56:184, 56:184, 13:141]

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(combined_x[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(combined_x[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1GD')
plt.subplot(223)
plt.imshow(combined_x[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')

plt.savefig("random_mri_slices_smaller.png")

# Save the array
np.save(TRAIN_DATASET_PATH + 'UNET_Train_PATH/combined00024.npy', combined_x)

# Verify image is being read properly
my_img = np.load(TRAIN_DATASET_PATH + 'UNET_Train_PATH/combined00024.npy')
print(f"✅ Loaded shape: {my_img.shape}")
print(f"📏 Size in MB: {my_img.nbytes / (1024 ** 2):.2f} MB")

# One-hot encode the mask
test_mask = to_categorical(test_mask, num_classes=4)
