'''
The purpose of this script is to evaluate the 8 trained models on the test dataset and print the results.

I will be performing the following steps:
- Importing necessary libraries and setting random seed for reproducibility
- Setting paths and directories.
- Getting list of patients that will be used for testing (must have mask.npy).
- Setting prediction parameters.
- Evaluating Mean IoU and Accuracy for each model on the testing set.
- Visualising predictions on a random patient slice for multiple modalities.
- Saving the visualisation results in the results directory.

# will need to change this to the test dataset path once i know this works
'''

#importing necessary libraries
import os
import re
import numpy as np
from keras.models import load_model
from keras.metrics import MeanIoU
import matplotlib.pyplot as plt
import random
import tensorflow as tf

# Setting random seed for reproducibility
'''
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
'''

# Setting paths and directories
test_dir = '/user/home/rb21991/DIS/Dissertation/data/Test_data/'
model_dir = 'saved_models'

# Getting list of patients that will be used for testing (must have mask.npy) ######################

test_patients_all   = sorted(os.listdir(test_dir))
test_mask_list = []
valid_test_patients = []

# Filter testing patients
for patient in test_patients_all:
    mask_path = os.path.join(test_dir, patient, 'mask.npy')
    if os.path.exists(mask_path):
        test_mask_list.append(mask_path)
        valid_test_patients.append(patient)
    else:
        print(f"Warning: mask.npy not found for testing patient {patient}")


test_patients   = valid_test_patients
print(f"Filtered {len(test_patients)} testing patients.")


test_patient_list = test_patients

# Loading data generators

# Creating data generator dictonaries #################################

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

# Prediction parameters  ###########################################################
batch_size = 4
n_classes = 4

best_iou = 0
best_model_name = None

best_accuracy = 0
best_model_name_accuracy = None
best_iou_for_best_accuracy = 0

# Mean IOU and Accuracy Evaluation #################################################

for model_file in os.listdir(model_dir):
    if not (model_file.endswith('.hdf5') or model_file.endswith('.keras')):
        continue

    model_path = os.path.join(model_dir, model_file)
    print(f"\nLoading model: {model_file}")

    # Extract modality indices from filename
    modality_indices = list(map(int, re.findall(r'\d+', model_file)))
    print(f"Using modality indices: {modality_indices}")

    # Load model
    model = load_model(model_path, compile=False)

    # Re-init the generator
    test_img_datagen = imageLoader(test_dir, test_patient_list, batch_size, modality_indices)

    # Set up metrics
    iou_metric = MeanIoU(num_classes=n_classes)
    total_correct = 0
    total_pixels = 0

    num_batches = len(test_patient_list) // batch_size

    for _ in range(num_batches):
        test_image_batch, test_mask_batch = next(test_img_datagen)
        test_mask_batch_argmax = np.argmax(test_mask_batch, axis=-1)

        test_pred_batch = model.predict(test_image_batch)
        test_pred_batch_argmax = np.argmax(test_pred_batch, axis=-1)

        # Update IoU
        iou_metric.update_state(test_mask_batch_argmax.flatten(), test_pred_batch_argmax.flatten())

        # Update accuracy
        total_correct += np.sum(test_mask_batch_argmax == test_pred_batch_argmax)
        total_pixels += test_mask_batch_argmax.size

    # Final metrics per model
    mean_iou = iou_metric.result().numpy()
    avg_accuracy = total_correct / total_pixels

    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")

    # Track best model
    if mean_iou > best_iou:
        best_iou = mean_iou
        best_model_name = model_file

    if avg_accuracy > best_accuracy:
        best_accuracy = avg_accuracy
        best_model_name_accuracy = model_file

print(f"\nBest model: {best_model_name} with Mean IoU: {best_iou:.4f}")
print(f"Model with Best Average Accuracy: {best_model_name_accuracy} with Average Accuracy: {best_accuracy:.4f})")



