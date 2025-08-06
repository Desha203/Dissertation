# https://youtu.be/ScdCQqLtnis
"""

Please get the data ready and define custom data generator using the other
files in this directory.

Images are expected to be 128x128x128x3 npy data (3 corresponds to the 3 channels for 
                                                  test_image_flair, test_image_t1ce, test_image_t2)
Change the U-net input shape based on your input dataset shape (e.g. if you decide to only se 2 channels or all 4 channels)

Masks are expected to be 128x128x128x3 npy data (4 corresponds to the 4 classes / labels)


You can change input image sizes to customize for your computing resources.
"""


import os
import numpy as np
from creating_data_batches import imageLoader
#import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random


####################################################
train_img_dir = "/user/home/rb21991/DIS/Dissertation/data/Train_data/Uimages/"
train_mask_dir = "/user/home/rb21991/DIS/Dissertation/data/Train_data/Umasks/"

img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

num_images = len(os.listdir(train_img_dir))

img_num = random.randint(0,num_images-1)
test_img = np.load(train_img_dir+img_list[img_num])
test_mask = np.load(train_mask_dir+msk_list[img_num])
test_mask = np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')

plt.tight_layout()
plt.savefig('SSSegmentation_visualization.png')

#just checking things 
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

#############################################################
#Optional step of finding the distribution of each class and calculating appropriate weights
#Alternatively you can just assign equal weights and see how well the model performs: 0.25, 0.25, 0.25, 0.25

import pandas as pd
columns = ['0','1', '2', '3']
df = pd.DataFrame(columns=columns)

train_mask_list = sorted(glob.glob('/user/home/rb21991/DIS/Dissertation/data/Train_data/Umasks/*.npy'))

print(len(train_mask_list))

for img in range(len(train_mask_list)):
    #print(img)
    temp_image=np.load(train_mask_list[img])
    temp_image = np.argmax(temp_image, axis=3)
    val, counts = np.unique(temp_image, return_counts=True)
    zipped = zip(columns, counts)
    conts_dict = dict(zipped)
    df = pd.concat([df, pd.DataFrame([conts_dict])], ignore_index=True)

    

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['1'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4
#Class weights claculation: n_samples / (n_classes * n_samples_for_class)
wt0 = round((total_labels/(n_classes*label_0)), 2) #round to 2 decimals
wt1 = round((total_labels/(n_classes*label_1)), 2)
wt2 = round((total_labels/(n_classes*label_2)), 2)
wt3 = round((total_labels/(n_classes*label_3)), 2)

print("Class weights: ", wt0, wt1, wt2, wt3)
#wt0, wt1, wt2, wt3 =  0.25 48.4 48.4 41.23 for 20 
#These weights can be used for Dice loss

##############################################################
#Define the image generators for training and validation

train_img_dir = "/user/home/rb21991/DIS/Dissertation/data/Train_data/Uimages/"
train_mask_dir = "/user/home/rb21991/DIS/Dissertation/data/Train_data/Umasks/"

val_img_dir = "/user/home/rb21991/DIS/Dissertation/data/Validate_data/Uimages/"
val_mask_dir = "/user/home/rb21991/DIS/Dissertation/data/Validate_data/Umasks/"

train_img_list=os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

val_img_list=os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)
#######################################

########################################################################
batch_size = 8

train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

print("image type: float64 (2, 128, 128, 128, 3)")
print("mask type: float32 (2, 128, 128, 128, 4)")

#Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))
print(test_img.shape)
print(f"Image shape: {img.shape}, dtype: {img.dtype}, range: {img.min()} - {img.max()}")
print(f"Mask shape: {msk.shape}, dtype: {msk.dtype}, range: {msk.min()} - {msk.max()}")

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1GD')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.tight_layout()
plt.savefig('checking_generator.png')


###########################################################################
# this is where i need to experiment  to gain understanding of doing somehing to dice function 

#Define loss, metrics and optimizer to be used for training
wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25

#from tensorflow.keras.optimizers import Adam


import segmentation_models_3D as sm
#dice and focal loss is from this library  

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = keras.optimizers.Adam(LR)


#######################################################################
#Fit the model 

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size


from  simple_3d_unet import simple_unet_model

model = simple_unet_model(IMG_HEIGHT=128, 
                          IMG_WIDTH=128, 
                          IMG_DEPTH=128, 
                          IMG_CHANNELS=3, 
                          num_classes=4)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)

#print(model.summary())

print(model.input_shape)
print('should be none,128,128,3 and 3 becomes 4 for mask' )
print(model.output_shape)


history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=100,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )


print('hello')
model.save('Unet3d.hdf5')
'''

##################################################################


#plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_validation_loss.png')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_validation_accuracy.png') 
'''

#################################################

from keras.models import load_model

#Load model for prediction or continue training

#For continuing training....
#The following gives an error: Unknown loss function: dice_loss_plus_1focal_loss
#This is because the model does not save loss function and metrics. So to compile and 
#continue training we need to provide these as custom_objects.

#my_model = load_model('Unet3d_100epochs_simple_unet_weighted_dice.hdf5')
my_model = load_model('Unet3d.hdf5',compile = False) # if you dont want to retrain the model
#So let us add the loss as custom object... but the following throws another error...
#Unknown metric function: iou_score
#my_model = load_model('Unet3d_100epochs_simple_unet_weighted_dice.hdf5', custom_objects={'dice_loss_plus_1focal_loss': total_loss})
'''
#when i am ready to retrain my model 
#Now, let us add the iou_score function we used during our initial training
my_model = load_model('saved_models/brats_3d_100epochs_simple_unet_weighted_dice.hdf5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)})

#Now all set to continue the training process. 
history2=my_model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=1,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )

'''
#################################################

#Verify IoU on a batch of images from the test dataset
#Using built in keras function for IU
#Only works on TF > 2.0 
#evauluating IoU on a batch of images

from keras.metrics import MeanIoU

batch_size=4 #Check IoU for a batch of images
test_img_datagen = imageLoader(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch, test_mask_batch = test_img_datagen.__next__()

print(f"test image type: {test_image_batch.dtype} {test_image_batch.shape}")
print(f"test mask type: {test_mask_batch.dtype} {test_mask_batch.shape}")

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

#############################################
#Predict on a few test images, one at a time USE VALIDATION SET
#Try images: 
img_num =3

test_img = np.load( val_img_dir +'image_' + str(img_num)+'.npy')

test_mask = np.load( val_mask_dir +'mask_' + str(img_num)+'.npy')

test_mask_argmax=np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]


print(test_prediction_argmax.shape)
print(test_mask_argmax.shape)
print(np.unique(test_prediction_argmax))


#Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

#n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice = 60
plt.figure(figsize=(12, 8))
print(f"test image type: {test_image_batch.dtype} {test_image_batch.shape}")
print(f"test mask type: {test_mask_batch.dtype} {test_mask_batch.shape}")

plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:,:,n_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax[:,:, n_slice])
plt.tight_layout()
plt.savefig('test_prediction_visualisation.png')

print('doneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
############################################################
