'''
The purpose of this script is to evaluate the 8 trained models on the test dataset and print the results.

I will be performing the following steps:
- Importing necessary libraries and seetting random seed for reproducibility.
- Setting paths and directories.
- Getting list of patients that will be used for validation (must have mask.npy).
- Setting prediction parameters.
- Evaluating Mean IoU and Accuracy for each model on the validation set.
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

from Training_Models import load_modalities
from Training_Models import load_mask
from Training_Models import imageLoader

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
val_dir = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'
model_dir = 'saved_models'

# Getting list of patients that will be used for validation (must have mask.npy) ######################

val_patients_all   = sorted(os.listdir(val_dir))
val_mask_list = []
valid_val_patients = []

# Filter validation patients
for patient in val_patients_all:
    mask_path = os.path.join(val_dir, patient, 'mask.npy')
    if os.path.exists(mask_path):
        val_mask_list.append(mask_path)
        valid_val_patients.append(patient)
    else:
        print(f"Warning: mask.npy not found for validation patient {patient}")


val_patients   = valid_val_patients
print(f"Filtered {len(val_patients)} validation patients.")


val_patient_list = val_patients


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

    # Extract modality indices from filename, e.g. "model_2_1.keras" -> [2,1]
    modality_indices = list(map(int, re.findall(r'\d+', model_file)))
    print(f"Using modality indices: {modality_indices}")

    # Create a new test generator with these modalities
    test_img_datagen = imageLoader(val_dir, val_patient_list, batch_size, modality_indices)

    # Load the model
    model = load_model(model_path, compile=False)

    # Evaluate the model on the entire validation set
    total_iou = 0
    total_accuracy = 0
    num_batches = len(val_patient_list) // batch_size

    
    # Reset the generator for each model evaluation
    test_img_datagen = imageLoader(val_dir, val_patient_list, batch_size, modality_indices)


    for i in range(num_batches):
        test_image_batch, test_mask_batch = next(test_img_datagen)
        test_mask_batch_argmax = np.argmax(test_mask_batch, axis=-1)

        test_pred_batch = model.predict(test_image_batch)
        test_pred_batch_argmax = np.argmax(test_pred_batch, axis=-1)

        # Calculate Mean IoU
        iou_metric = MeanIoU(num_classes=n_classes)
        iou_metric.update_state(test_mask_batch_argmax.flatten(), test_pred_batch_argmax.flatten())
        mean_iou = iou_metric.result().numpy()
        print(f"Mean IoU: {mean_iou:.4f}")

        # Calculate Accuracy for the current batch
        correct_pixels = np.sum(test_mask_batch_argmax == test_pred_batch_argmax)
        total_pixels = test_mask_batch_argmax.size
        batch_accuracy = correct_pixels / total_pixels
        total_accuracy += batch_accuracy


avg_accuracy = total_accuracy / num_batches

print(f"  Average Accuracy: {avg_accuracy:.4f}")

if mean_iou > best_iou:
    best_iou = mean_iou
    best_model_name = model_file

if avg_accuracy > best_accuracy:
    best_accuracy = avg_accuracy
    best_model_name_accuracy = model_file
   

print(f"\nBest model: {best_model_name} with Mean IoU: {best_iou:.4f}")
print(f"Model with Best Average Accuracy: {best_model_name_accuracy} with Average Accuracy: {best_accuracy:.4f} )")

# Visualisation of Predictions on a Random Patient Slice Multiple Modalities ##########################

# Get a random patient and slice for visualisation
n_slice = 43  
patients = val_patients
random_patient = random.choice(patients)
print(f"Randomly selected patient for visualisation: {random_patient}")
patient_path = f"../data/Validate_data/{random_patient}/"


if not val_patient_list:
    print(f"Error: No patient folders found in {val_dir}")
else:
    # Randomly select a patient to visualize
    patient_to_visualize = random.choice(val_patient_list)
    print(f"Randomly selected patient for visualization: {patient_to_visualize}")

    patient_path = os.path.join(val_dir, patient_to_visualize)

    if not os.path.exists(patient_path):
        print(f"Error: Patient folder not found at {patient_path}")
    else:
        # Get a random slice index for visualization
        # Assuming all volumes have the same depth after preprocessing
        # Load one of the preprocessed .npy files to get the depth
        sample_npy_path = os.path.join(patient_path, 't1.npy')
        if os.path.exists(sample_npy_path):
            sample_volume = np.load(sample_npy_path)
            volume_depth = sample_volume.shape[2]
            n_slice = 83
            print(f"Visualizing slice {n_slice} for {patient_to_visualize}")

            # Load the ground truth mask once
            mask = np.load(os.path.join(patient_path, "mask.npy"))  # shape: (128,128,128, 4)
            mask_argmax = np.argmax(mask, axis=-1) # Convert mask to labels

            # Get the list of models to visualize
            model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.keras')])
            num_models = len(model_files)

            if num_models == 0:
                print("No .keras models found in the saved_models directory.")
            else:
                # Create a single figure for side-by-side comparison
                # We need 2 + num_models columns: 1 for input, 1 for GT mask, and num_models for predictions
                plt.figure(figsize=(5 * (2 + num_models), 6))
                plt.suptitle(f"Predictions for {patient_to_visualize} (Slice {n_slice})", fontsize=16)

                # Load and plot the first modality of the first model as the common input
                if len(model_files) > 0:
                     first_model_file = model_files[0]
                     combo_str = first_model_file.replace("model_", "").replace(".keras", "")
                     modality_indices = list(map(int, combo_str.split('_')))
                     modality_names = ['t1', 't1GD', 't2', 'flair']
                     selected_modalities = [modality_names[i] for i in modality_indices]
                     if selected_modalities:
                         first_modality_path = os.path.join(patient_path, f"{selected_modalities[0]}.npy")
                         if os.path.exists(first_modality_path):
                              first_modality_image = np.load(first_modality_path)
                              plt.subplot(1, 2 + num_models, 1)
                              plt.title(f"Input ({selected_modalities[0].upper()})")
                              plt.imshow(first_modality_image[:, :, n_slice], cmap='gray')
                              plt.axis('off')
                         else:
                              print(f"Warning: First modality {selected_modalities[0]}.npy not found for patient {patient_to_visualize}. Skipping input plot.")


                # Plot Ground Truth Mask
                plt.subplot(1, 2 + num_models, 2)
                plt.imshow(mask_argmax[:, :, n_slice])
                plt.title("Ground Truth Mask")
                plt.axis('off')

                # Loop through models and plot predictions
                for i, model_file in enumerate(model_files):
                    print(f"\nLoading model: {model_file}")
                    model_path = os.path.join(model_dir, model_file)

                    try:
                        # Load the model
                        model = load_model(model_path, compile=False)

                        # Extract modality indices from filename
                        combo_str = model_file.replace("model_", "").replace(".keras", "")
                        modality_indices = list(map(int, combo_str.split('_')))
                        modality_names = ['t1', 't1GD', 't2', 'flair']
                        selected_modalities = [modality_names[i] for i in modality_indices]

                        # Load selected modalities for the current patient
                        modality_arrays = []
                        for mod in selected_modalities:
                            modality_path = os.path.join(patient_path, f"{mod}.npy")
                            if os.path.exists(modality_path):
                                modality_arrays.append(np.load(modality_path))
                            else:
                                print(f"Warning: Modality {mod}.npy not found for patient {patient_to_visualize}. Skipping prediction for this model.")
                                modality_arrays = None # Indicate missing modalities
                                break

                        if modality_arrays is not None and len(modality_arrays) > 0:
                            stacked_input = np.stack(modality_arrays, axis=-1)  # shape: (128,128,128,len(combo))
                            test_input = np.expand_dims(stacked_input, axis=0)  # shape: (1,128,128,128,channels)

                            # Predict
                            prediction = model.predict(test_input)
                            prediction_argmax = np.argmax(prediction, axis=-1)[0]  # shape: (128,128,128)

                            # Plot Prediction
                            plt.subplot(1, 2 + num_models, i + 3) # Start plotting predictions after Input and GT Mask
                            plt.title(f"Pred ({'_'.join(selected_modalities)})")
                            plt.imshow(prediction_argmax[:, :, n_slice])
                            plt.axis('off')

                    except Exception as e:
                        print(f"Error loading or processing model {model_file}: {e}")

                os.makedirs("../results", exist_ok=True)
                save_path = os.path.join("../results", f"all_models_prediction_{patient_to_visualize}_slice{n_slice}.png")
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches='tight')
                print(f"\nSaved side-by-side visualisation to: {save_path}")
                plt.show()
                plt.close()

        else:
            print(f"Error: Sample .npy file not found at {sample_npy_path}")