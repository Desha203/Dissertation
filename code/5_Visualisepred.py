'''
The purpose of this script is to visualise the predictions of multiple models on a randomly selected patient from the test dataset.
I will be performing the following steps:
- import necessary libraries
- set random seed for reproducibility
- set paths and directories
- get the list of test patients
- randomly select a patient to visualise
- load the models and visualise predictions for selected slices
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

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# Setting paths and directories
test_dir = '/user/home/rb21991/DIS/Dissertation/data/Test_data/'
model_dir = 'saved_models'

# Getting list of patients that will be used for test (must have mask.npy) ######################

test_patients_all   = sorted(os.listdir(test_dir))
test_mask_list = []
valid_test_patients = []

# Filter test patients
for patient in test_patients_all:
    mask_path = os.path.join(test_dir, patient, 'mask.npy')
    if os.path.exists(mask_path):
        test_mask_list.append(mask_path)
        valid_test_patients.append(patient)
    else:
        print(f"Warning: mask.npy not found for test patient {patient}")


test_patients   = valid_test_patients
#print(f"Filtered {len(val_patients)} validation patients.")


test_patient_list = test_patients
print("Test patients:", len(test_patient_list))


# visualise predictions for a randomly selected patient

if not test_patient_list:
    print(f"Error: No patient folders found in {test_dir}")
else:
    # Select patient 
    selected_patients = ['UPENN-GBM-00060_11', 'UPENN-GBM-00014_11', 'UPENN-GBM-00151_11']

    for patient_to_visualize in selected_patients:
        print(f"Randomly selected patient for visualisation: {patient_to_visualize}")

        patient_path = os.path.join(test_dir, patient_to_visualize)

        if not os.path.exists(patient_path):
            print(f"Error: Patient folder not found at {patient_path}")
        else:
            # Define a list of slices to visualise
            #slices_to_visualize = [30, 40, 50, 59, 60, 65, 70, 80, 90, 100] # Example slice indices you can edit
            #slices_to_visualize = [20,25,30,35,40,45,46,48]
            slices_to_visualise = [10, 20, 30, 35, 45, 55, 65, 75, 83, 90, 100, 110, 120]


            # Assuming all volumes have the same depth after preprocessing
            # Load one of the preprocessed .npy files to get the depth
            sample_npy_path = os.path.join(patient_path, 't1.npy')
            if os.path.exists(sample_npy_path):
                sample_volume = np.load(sample_npy_path)
                volume_depth = sample_volume.shape[2]
                print(f"Volume depth for patient {patient_to_visualize}: {volume_depth}")


                # Load the ground truth mask once
                mask = np.load(os.path.join(patient_path, "mask.npy"))  # shape: (128,128,128, 4)
                mask_argmax = np.argmax(mask, axis=-1) # Convert mask to labels

                # Get the list of models to visualize
                model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.keras')])
                num_models = len(model_files)

                if num_models == 0:
                    print("No .keras models found in the saved_models directory.")
                else:
                    for n_slice in slices_to_visualise:
                        if n_slice >= volume_depth:
                            print(f"Warning: Slice {n_slice} is out of bounds for volume depth {volume_depth}. Skipping this slice.")
                            continue

                        # Create a single figure for side-by-side comparison
                        plt.figure(figsize=(20, 12)) # Adjusted figure size for 2x5 grid
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
                                    plt.subplot(2, 5, 1) # Position 1 in 2x5 grid
                                    plt.title(f"Input ({selected_modalities[0].upper()})")
                                    plt.imshow(first_modality_image[:, :, n_slice], cmap='gray')
                                    #plt.axis('off') # Removed to keep axis
                                else:
                                    print(f"Warning: First modality {selected_modalities[0]}.npy not found for patient {patient_to_visualize}. Skipping input plot.")


                        # Plot Ground Truth Mask
                        plt.subplot(2, 5, 2) # Position 2 in 2x5 grid
                        plt.imshow(mask_argmax[:, :, n_slice])
                        plt.title("Ground Truth Mask")
                        #plt.axis('off') # Removed to keep axis

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
                                    plt.subplot(2, 5, i + 3) # Positions 3 to 10 in 2x5 grid
                                    plt.title(f"Pred ({'_'.join(selected_modalities)})")
                                    plt.imshow(prediction_argmax[:, :, n_slice])
                                    #plt.axis('off') # Removed to keep axis

                            except Exception as e:
                                print(f"Error loading or processing model {model_file}: {e}")

                        # Create a folder for each patient inside ../results
                        results_dir = os.path.join("../results", f"{patient_to_visualize}_predictions")
                        os.makedirs(results_dir, exist_ok=True)
                        save_path = os.path.join(results_dir, f"slice{n_slice}.png")
                        plt.tight_layout()
                        plt.savefig(save_path, bbox_inches='tight')
                        print(f"\nSaved side-by-side visualisation to: {save_path}")
                        plt.show()
                        plt.close()

            else:
                print(f"Error: Sample .npy file not found at {sample_npy_path}")

