# uploading data

import os
import shutil
import random
from pathlib import Path

'''This script is designed to run in Google Colab, where it mounts Google Drive to access files.
from google.colab import drive
drive.mount('/content/drive')



# Change to correct directory
%cd "/content/drive/MyDrive/Dissertation"

# Show current path and contents
!pwd
!ls
os.chdir("/content/drive/MyDrive/Dissertation")
print("Current working directory:", Path.cwd())

''' 
#will comment for hpc


import os
import shutil
import random
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


import tensorflow as tf  # or torch if you're using PyTorch

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Making directories
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

'''
# --- CONFIGURATION ---
source_dir = Path("/content/drive/MyDrive/Dissertation/Segmentation_Masks")  # Corrected path
subject_dir = Path("/content/drive/MyDrive/Dissertation/Mri_scans_images")   # Corrected path
output_base = Path("data")

'''
# FOR HPC

source_dir = Path("/user/home/ms13525/scratch/hackathon-2025/data/images_segm")  # Directory with the NIfTI files
subject_dir = Path("/user/home/ms13525/scratch/hackathon-2025/data/images_structural")  # Directory with folders for each subject
output_base = Path("../data")

'''
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


# Ensure at least 1 file in each if total_files allows  # NEXT TWO IFS NEED TO BE REMOVE WHEN MOVED TO HPC!
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
            print(f" Moved {filename} → {subject_folder}")
        else:
            print(f"Folder not found for {filename}: expected {subject_folder}")

print(" Done: All files organised and split.")

######################################### Images to numpy ############################################################################


# Base path: resolve relative to current script location
#base_path = Path.cwd()
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#from tensorflow.keras.utils import to_categorical # Import to_categorical

'''# Define paths for training and validation datasets''
TRAIN_DATASET_PATH = 'data/Train_data'
VALIDATION_DATASET_PATH = 'data/Validate_data'

'''
#For HPC

TRAIN_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
VALIDATION_DATASET_PATH = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'



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


    
    # Remove the original .nii.gz files MIGHT DO LATER
   # os.remove(t1_path)
    #os.remove(t2_path)
    #os.remove(t1ce_path)
    #os.remove(flair_path)
    #os.remove(mask_path)
    

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


    
    # Remove the original .nii.gz files MIGHT DO LATER
    #os.remove(t1_path)
    #os.remove(t2_path)
    #os.remove(t1ce_path)
    #os.remove(flair_path)
    #os.remove(mask_path)
    

    print(f"✔ Finished and replaced .nii.gz files for validation {patient_id}")


    ########################################### CHECK#####################

import os
import numpy as np
#need to check the >1% usefullness
# double checking
patient_id = "UPENN-GBM-00006_11" # will need to edit
base_path = os.path.join("../data/Train_data", patient_id)

# Load modalities
modalities = ['t1.npy', 't2.npy', 't1GD.npy', 'flair.npy']
images = [np.load(os.path.join(base_path, m)) for m in modalities]
mask = np.load(os.path.join(base_path, "mask.npy"))

# Check and print for each modality
for name, img in zip(modalities, images):
    print(f"--- {name.upper()} ---")
    print("Shape:", img.shape)
    print("Data type MRI scan:", img.dtype)
    print("Intensity range:", round(img.min(), 4), "-", round(img.max(), 4))
    print()

# Check mask
labels, counts = np.unique(mask, return_counts=True)
print("--- MASK ---")
print("Mask shape:", mask.shape)
print("Mask labels and counts:", dict(zip(labels, counts)))
print("Mask dtype:", mask.dtype)

# Check if mask is one-hot encoded
if mask.ndim == 4 and mask.shape[-1] == 4:
    print("Mask appears to be one-hot encoded.")
else:
    print("Mask is not one-hot encoded. Shape or dimensions incorrect.")

'''
######################### Make and check data generators ###########################################


def load_modalities(patient_dir, modality_indices): # goes through each of the patient folders and stack the modalities that i need for a give model.
    modality_map = {
        0: 't1.npy',
        1: 't1GD.npy',
        2: 't2.npy',
        3: 'flair.npy'
    }

    modalities = []
    for idx in modality_indices:
        modality_file = modality_map[idx]
        modality_path = os.path.join(patient_dir, modality_file)
        data = np.load(modality_path)
        modalities.append(data)

    return np.stack(modalities, axis=0)  # Shape: (C, 128, 128, 128)

def load_mask(patient_dir):
    return np.load(os.path.join(patient_dir, 'mask.npy'))

def imageLoader(data_dir, patient_list, batch_size, modality_indices):
    L = len(patient_list)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            patients = patient_list[batch_start:limit]

            X_batch = []
            Y_batch = []

            for patient in patients:
                patient_dir = os.path.join(data_dir, patient)

                # Load modalities: shape (C, 128, 128, 128)
                X = load_modalities(patient_dir, modality_indices)

                # Load mask: shape (128, 128, 128)
                Y = load_mask(patient_dir).astype(np.uint8)  # shape: (128,128,128)
                #print(Y.shape)


                # Transpose X to (128, 128, 128, C) for TensorFlow
                X = np.transpose(X, (1, 2, 3, 0))

                X_batch.append(X)
                Y_batch.append(Y)

            yield np.array(X_batch, dtype=np.float32), np.array(Y_batch, dtype=np.float32)

            batch_start += batch_size
            batch_end += batch_size



import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Set paths
#train_dir = TRAIN_DATASET_PATH
train_dir = '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
patient_list = sorted(os.listdir(train_dir))  # list of patient folders

# Example: use T2 modality only (index 2)
modality_indices = [2,1,3,0]
batch_size = 1

# Create generator
train_gen = imageLoader(train_dir, patient_list, batch_size, modality_indices)

# Get one batch
images, masks = next(train_gen)

# Pick a random image in the batch
img_idx = random.randint(0, images.shape[0] - 1)
image = images[img_idx]     # shape: [128, 128, 128, C]
mask = masks[img_idx]       # shape: [128, 128, 128, 1]
print("Images shape:", image.shape)
print("Masks shape:", mask.shape)


# Pick a random axial slice
n_slice = random.randint(0, 127)

# Plot
plt.figure(figsize=(10, 5))

# Plot selected modality channels (C could be 1 or more)
num_modalities = image.shape[-1]
for i in range(num_modalities):
    plt.subplot(2, num_modalities, i + 1)
    plt.imshow(image[:, :, n_slice, i], cmap='gray')
    plt.title(f'Modality {modality_indices[i]}')

# Plot mask
plt.subplot(2, num_modalities, num_modalities + 1)
plt.imshow(np.argmax(mask[:, :, n_slice, :], axis=-1))
plt.title('Mask')


# ALL GOOD BUT I might decide not to do the cropping anymore
print("Example Proccessed data set:")
plt.tight_layout()
plt.show()
plt.savefig("example_generator_data.png")


################################### Load 3d Unet ##############################################

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU

kernel_initializer =  'he_uniform' #Try others if you want

################################################################
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    #s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    #Contraction path
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)

    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)

    #Expansive path
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)

    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)

    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)

    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible.
    #model.summary()

    return model

#Test if everything is working ok.
model = simple_unet_model(128, 128, 128, 3, 4)
print(model.input_shape)
print(model.output_shape)


#################### enquire about masks ###################################

# Paths
#train_dir = TRAIN_DATASET_PATH
#val_dir   = VALIDATION_DATASET_PATH

train_dir =  '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
val_dir = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'
model_dir  = "saved_models"

os.makedirs(model_dir, exist_ok=True)

# Patient folders
train_patients = sorted(os.listdir(train_dir))
val_patients   = sorted(os.listdir(val_dir))

#################################################################################################################################################

import pandas as pd

columns = ['0', '1', '2', '3']
df = pd.DataFrame(columns=columns)

#train_dataset_path = TRAIN_DATASET_PATH  # Update if different
train_dataset_path = '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
train_patients = sorted(os.listdir(train_dataset_path))

print(f"Found {len(train_patients)} patient folders.")

columns = ['0','1', '2', '3']
df = pd.DataFrame(columns=columns)

train_mask_list = []

for patient in train_patients:
    mask_path = os.path.join(train_dataset_path, patient, 'mask.npy')
    if os.path.exists(mask_path):
        train_mask_list.append(mask_path)
    else:
        print(f"Warning: mask.npy not found for patient {patient}")

import numpy as np
import pandas as pd

# Assuming train_mask_list is a sorted list of file paths to your masks
train_mask_list = sorted(train_mask_list)

import glob
import numpy as np
import pandas as pd

columns = ['0', '1', '2', '3']
df = pd.DataFrame(columns=columns)


for img_idx, mask_path in enumerate(train_mask_list):
    print(img_idx)
    temp_image = np.load(mask_path)
    temp_image = np.argmax(temp_image, axis=3)

    val, counts = np.unique(temp_image, return_counts=True)

    conts_dict = {c: counts[val.tolist().index(int(c))] if int(c) in val else 0 for c in columns}

    df = pd.concat([df, pd.DataFrame([conts_dict])], ignore_index=True)

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['2'].sum()
label_3 = df['3'].sum()

total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4

# Raw class weights (no normalisation)
wt0 = round(total_labels / (n_classes * label_0), 2)
wt1 = round(total_labels / (n_classes * label_1), 2)
wt2 = round(total_labels / (n_classes * label_2), 2)
wt3 = round(total_labels / (n_classes * label_3), 2)

print("Class weights:", wt0, wt1, wt2, wt3)


################################## training scripts ###################################

import os
#from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
#from creating_data_batches import imageLoader
#import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random
from tensorflow.keras.callbacks import ModelCheckpoint


#just checking things
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define all 8 combinations that include T2 (index 2) ONLY as focussing on ED region
modality_combos = [ [1, 2, 3] ]

# Paths
train_dir = TRAIN_DATASET_PATH
val_dir   = VALIDATION_DATASET_PATH

train_dir = '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
val_dir = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'
model_dir  = "saved_models"

os.makedirs(model_dir, exist_ok=True)

# Patient folders USED IN TRAINING

import os

# Paths
train_dir = '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
val_dir   = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'

# Model save path
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# Get all patient folders
train_patients_all = sorted(os.listdir(train_dir))
val_patients_all   = sorted(os.listdir(val_dir))

# Init lists
train_mask_list = []
valid_train_patients = []

val_mask_list = []
valid_val_patients = []

# Filter training patients
for patient in train_patients_all:
    mask_path = os.path.join(train_dir, patient, 'mask.npy')
    if os.path.exists(mask_path):
        train_mask_list.append(mask_path)
        valid_train_patients.append(patient)
    else:
        print(f"Warning: mask.npy not found for training patient {patient}")

# Filter validation patients
for patient in val_patients_all:
    mask_path = os.path.join(val_dir, patient, 'mask.npy')
    if os.path.exists(mask_path):
        val_mask_list.append(mask_path)
        valid_val_patients.append(patient)
    else:
        print(f"Warning: mask.npy not found for validation patient {patient}")

# Final filtered lists
train_patients = valid_train_patients
val_patients   = valid_val_patients

'''
train_patients = sorted(os.listdir(train_dir))
val_patients   = sorted(os.listdir(val_dir))
'''
#############################################################################

# this is where i need to experiment to gain understanding of which weighting best highlights the ED section if that is the project idea I got with

#Define loss, metrics and optimizer to be used for training
wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25

#from tensorflow.keras.optimizers import Adam

import segmentation_models_3D as sm

#dice and focal loss is from this library

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  # will explain in dissertation why I combined this loss function

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = keras.optimizers.Adam(LR)


# Training parameters
batch_size = 4
epochs = 100  # You can increase this
steps_per_epoch = len(train_patients) // batch_size
val_steps = len(val_patients) // batch_size


########################################################### TRAIN MODELS ###################################################################
for combo in modality_combos:
    print(f"Training model with modalities: {combo}")

    # Check if all required .npy files exist for all patients in train and val
   
    # Create generators
    train_gen = imageLoader(train_dir, train_patients, batch_size, combo)
    val_gen = imageLoader(val_dir, val_patients, batch_size, combo)

    # Get one batch from train_gen to inspect shapes
    images, masks = next(train_gen)
    print(f"Images shape: {images.shape}")  # Expect (batch_size, 128,128,128, input_channels)
    print(f"Masks shape: {masks.shape}")    # Expect (batch_size, 128,128,128, 4) if one-hot

    # Build model
    input_channels = len(combo)
    model = simple_unet_model(128, 128, 128, input_channels, 4)

    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

     # Create a NEW optimizer instance here
    optim = tf.keras.optimizers.Adam()  # or whatever you were using

    model.compile(optimizer=optim, loss=total_loss, metrics=metrics)

    combo_str = '_'.join(map(str, combo))
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(model_dir, f"model_{combo_str}.keras"),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        initial_value_threshold=float('inf')

    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint]
    )

################################### evaluation and viz ############################################################


import os
import re
import numpy as np
from keras.models import load_model
from keras.metrics import MeanIoU
import matplotlib.pyplot as plt

model_dir = 'saved_models'
batch_size = 1
n_classes = 4
#val_dir = VALIDATION_DATASET_PATH
val_dir = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'
val_patient_list = sorted(os.listdir(val_dir))

best_iou = 0
best_model_name = None

for model_file in os.listdir(model_dir):
    if not (model_file.endswith('.hdf5') or model_file.endswith('.keras')):
        continue

    model_path = os.path.join(model_dir, model_file)
    print(f"\nLoading model: {model_file}")

    # Extract modality indices from filename, e.g. "model_2_1.keras" -> [2,1]
    modality_indices = list(map(int, re.findall(r'\d+', model_file)))
    print(f"Using modality indices: {modality_indices}")

    # Create a new test generator with these modalities
    test_img_datagen = imageLoader(val_dir, val_patient_list, batch_size, modality_indices)

    # Load the model
    model = load_model(model_path, compile=False)

    # Get one batch from test data
    test_image_batch, test_mask_batch = next(test_img_datagen)
    test_mask_batch_argmax = np.argmax(test_mask_batch, axis=-1)  # one-hot to labels

    # Predict
    test_pred_batch = model.predict(test_image_batch)
    test_pred_batch_argmax = np.argmax(test_pred_batch, axis=-1)

    # Calculate Mean IoU
    iou_metric = MeanIoU(num_classes=n_classes)
    iou_metric.update_state(test_mask_batch_argmax.flatten(), test_pred_batch_argmax.flatten())
    mean_iou = iou_metric.result().numpy()
    print(f"Mean IoU: {mean_iou:.4f}")

    if mean_iou > best_iou:
        best_iou = mean_iou
        best_model_name = model_file

print(f"\nBest model: {best_model_name} with Mean IoU: {best_iou:.4f}")


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import random

model_dir = 'saved_models'
#val_dir = VALIDATION_DATASET_PATH
val_dir= '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'
#modalities = ['t1.npy', 't1GD.npy', 't2.npy', 'flair.npy']
n_slice = 33
  # we'll stack modalities into channels, so channel index here
import os
import numpy as np
import random
from keras.models import load_model
import matplotlib.pyplot as plt


n_slice = 60  # slice index for visualisation

# Get a random patient
patients = sorted(os.listdir(val_dir))
random_patient = random.choice(patients)
patient_path = os.path.join(val_dir, random_patient)

# Loop through models #choose a slice from a random patient in validation sets and predicts it.
for model_file in os.listdir(model_dir):
    if not model_file.endswith('.keras'):
        continue

    print(f"\nLoading model: {model_file}")
    model_path = os.path.join(model_dir, model_file)
    model = load_model(model_path, compile=False)

    # Extract modality indices from filename
    combo_str = model_file.replace("model_", "").replace(".keras", "")
    modality_indices = list(map(int, combo_str.split('_')))
    modality_names = ['t1', 't1GD', 't2', 'flair']
    selected_modalities = [modality_names[i] for i in modality_indices]

    # Load modalities
    modality_arrays = []
    for mod in selected_modalities:
        modality_path = os.path.join(patient_path, f"{mod}.npy")
        modality = np.load(modality_path)
        modality_arrays.append(modality)
    stacked_input = np.stack(modality_arrays, axis=-1)  # shape: (128,128,128,len(combo))
    test_input = np.expand_dims(stacked_input, axis=0)  # shape: (1,128,128,128,channels)

    # Load and process mask
    mask = np.load(os.path.join(patient_path, "mask.npy"))  # shape: (128,128,128)

    # Predict
    prediction = model.predict(test_input)
    prediction_argmax = np.argmax(prediction, axis=-1)[0]  # shape: (128,128,128)

    # Visualise
    plt.figure(figsize=(12, 4))
    plt.suptitle(f"{model_file} on {random_patient}", fontsize=14)

    plt.subplot(1, 3, 1)
    plt.title("Input (first modality)")
    plt.imshow(stacked_input[:, :, n_slice, 0], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(np.argmax(mask[:, :, n_slice, :], axis=-1))
    plt.title("Ground Truth Mask")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(prediction_argmax[:, :, n_slice])

    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", f"{model_file.replace('.keras', '')}_slice{n_slice}.png")
    plt.savefig(save_path)
    print(f"Saved visualisation to: {save_path}")
    plt.close()
