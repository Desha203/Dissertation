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
import matplotlib.pyplot as plt


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# PART 1: Load and preprocess using 1 example #################################################################### 

TRAIN_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
VALIDATION_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'
#UNET_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/UNet_data/'

#####################################

#Now let us apply the same as above to all the images...
#Merge channels, crop, patchify, save
#GET DATA READY =  GENERATORS OR OTHERWISE

#Keras datagenerator does ntot support 3d

# # # images lists harley
#t1_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
t2_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*T2.nii*'))
t1GD_list = sorted(glob.glob(TRAIN_DATASET_PATH +'*/*T1GD.nii*'))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH +'*/*FLAIR.nii*'))
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH + '*/*segm.nii*'))

print(len(t2_list))
print(len(t1GD_list))
print(len(flair_list))
print(len(mask_list))

#Each volume generates 18 64x64x64x4 sub-volumes. 
#Total 369 volumes = 6642 sub volumes

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
    print(np.unique(temp_mask))
    
    
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1GD, temp_image_t2], axis=3)
    
    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
    #cropping x, y, and z
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]
    
    val, counts = np.unique(temp_mask, return_counts=True)
    
    if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume
     print("Save Me")
    
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
    
    

