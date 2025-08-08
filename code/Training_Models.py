'''
The purpose of this script is to create 8 3D U-Net models each trained on different modalities of MRI data.

I will be performing the following steps:
- Importing necessary libraries.
- Setting random seed for reproducibility and checking available GPUs.
- Defining the modalities combinations to be used for training.
- Creating data generators to load images and masks.
- Defining a simple 3D U-Net model architecture.    
- Setting class weights, loss function, callbacks, and training parameters.
- Training the models for each modality combination.
- Plotting training and validation loss and accuracy for each modality combination.
- Saving the trained models and plots.
'''

#import necessary libraries 
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU
import keras
import glob
import random
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import segmentation_models_3D as sm

#setting random seed for reproducibility and checking available GPUs
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# creating directories and setting paths  #########################################

train_dir = '/user/home/rb21991/DIS/Dissertation/data/Train_data/'
val_dir = '/user/home/rb21991/DIS/Dissertation/data/Validate_data/'
patient_list = sorted(os.listdir(train_dir))  # list of patient folders

model_dir  = "saved_models"
os.makedirs(model_dir, exist_ok=True)
os.makedirs('../results/training_plots', exist_ok=True)  # Create a directory for training plots if it doesn't exist

# Getting list of patients that will be used for training (must have mask.npy)

train_patients = sorted(os.listdir(train_dir))
val_patients   = sorted(os.listdir(val_dir))

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
print(f"Filtered {len(train_patients)} training patients and {len(val_patients)} validation patients.")


#Defining modalities combos ########################################

#Example: use T2 modality only (index 2)
modality_indices = [2,1,3,0]

modality_combos = [
    [2], [3], [1, 2], [2, 3],
    [1, 2, 3], [0, 1, 2], [0, 2, 3], [0, 1, 2, 3]
]


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

# Loading 3D U-Net model ##########################

kernel_initializer =  'he_uniform' #Try others if you want

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

# Setting class weights, loss function, callbacks and training parameters ##########################################

wt0, wt1, wt2, wt3 = 0.1, 0.5, 0.2, 0.2 # class weights for background, edema, etumor, necrosis

#Creating loss function using Dice and Focal loss

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  # will explain in dissertation why I combined this loss function

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = keras.optimizers.Adam(LR)

# Custom callback to track best epoch and training time
class BestEpochCallback(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', mode='min'):
        super(BestEpochCallback, self).__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.best_epoch = -1
        self.train_start_time = None
        self.train_end_time = None
        self.history = {} # Store history for plotting later

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if self.mode == 'min':
            if current_value < self.best_value:
                self.best_value = current_value
                self.best_epoch = epoch + 1  # Epochs are 0-indexed
        else: # mode == 'max'
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_epoch = epoch + 1

        # Store epoch-wise history
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)


    def on_train_end(self, logs=None):
        self.train_end_time = time.time()
        self.training_duration = self.train_end_time - self.train_start_time
        print(f"\nTraining finished.")
        print(f"Best Epoch ({self.monitor}): {self.best_epoch}")
        print(f"Training Duration: {self.training_duration:.2f} seconds")

# Training parameters
batch_size = 4
epochs = 100 # You can increase this
steps_per_epoch = len(train_patients) // batch_size
val_steps = len(val_patients) // batch_size

# Store history for plotting
histories = {}

######################### Training the models for each modality combination ##########################################

for combo in modality_combos:
    print(f"Training model with modalities: {combo}")

    # Check if all required .npy files exist for all patients in train and val
   
    # Create generators
    train_gen = imageLoader(train_dir, train_patients, batch_size, combo)
    val_gen = imageLoader(val_dir, val_patients, batch_size, combo)

    # Get one batch from train_gen to inspect shapes
    images, masks = next(train_gen)
    

    # Build model
    input_channels = len(combo)
    model = simple_unet_model(128, 128, 128, input_channels, 4)

    #print(f"Model input shape: {model.input_shape}")
    #print(f"Model output shape: {model.output_shape}")

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

    # Instantiate the custom callback
    best_epoch_callback = BestEpochCallback(monitor='val_loss', mode='min')


    history=model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint, best_epoch_callback] # Add the custom callback here
    )
    # Store history for plotting
    histories[combo_str] = history.history

    print(f"Finished training model with modalities: {combo}")

# Plotting training and validation loss for each modality combination

plt.figure(figsize=(12, 6))
for combo_str, history_dict in histories.items():
    epochs = range(1, len(history_dict['loss']) + 1)
   # plt.plot(epochs, history_dict['loss'], label=f'Training loss ({combo_str})')
    plt.plot(epochs, history_dict['val_loss'], label=f'Validation loss ({combo_str})')
plt.title(' Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'../results/training_plots/validation_loss_{combo_str}.png')  
plt.close()

# Plot and save training/validation accuracy
plt.figure(figsize=(12, 6))
for combo_str, history_dict in histories.items():
    epochs = range(1, len(history_dict['accuracy']) + 1)
    #plt.plot(epochs, history_dict['accuracy'], label=f'Training accuracy ({combo_str})')
    plt.plot(epochs, history_dict['val_accuracy'], label=f'Validation accuracy ({combo_str})')
plt.title(' Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'../results/training_plots/validation_accuracy_{combo_str}.png')  
plt.close()

print("Training complete. Models and plots saved.")
