#############################################
#Predict on a few test images, one at a time USE VALIDATION SET
val_img_dir = "../data/Validate_data/Uimages/"
val_mask_dir = "../data/Validate_data/Umasks/"


#Try images:
img_num =0

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
n_slice = 33
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
plt.savefig('prediction.png')

print('doneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
