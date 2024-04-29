import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense,Dropout,Activation,Flatten,Lambda
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from skimage.transform import rotate

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1
NUM_TEST_IMAGES = 20

images_list = os.listdir('../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_images/BBBC005_v1_images')
masks_list = os.listdir('../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth')

#Creating a data frame to understand data
df_img = pd.DataFrame(images_list, columns= ['image_id'])
df_img = df_img[df_img['image_id'] != '.htaccess']

print(df_img.head())


def GetNumberOfCells(x):
    a = x.split('_')
    b = a[2]
    num_cells = int(b[1:])

    return num_cells


def MaskExist(x):
    if x in masks_list:
        return 'yes'
    else:
        return 'no'


def HasBlur(x):
    a = x.split('_')
    b = a[3]
    blur_level = int(b[1:])

    return blur_level


df_img['num_cells'] = df_img['image_id'].apply(GetNumberOfCells)
df_img['has_mask'] = df_img['image_id'].apply(MaskExist)
df_img['blur_amt'] = df_img['image_id'].apply(HasBlur)


df_masks = df_img[df_img['has_mask']== 'yes']
df_masks.head()

df_test = df_masks.sample(NUM_TEST_IMAGES, random_state=101)
df_test

df_test = df_test.reset_index(drop=True)
test_images_list = list(df_test['image_id'])
df_masks = df_masks[~df_masks['image_id'].isin(test_images_list)]

print(df_masks.shape)
print(df_test.shape)


sample_image = df_img['image_id'][2]
path_image = '../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_images/BBBC005_v1_images/' + sample_image

# read the image using skimage
image = imread(path_image)
plt.imshow(image, cmap="gray")

print('Average pixel value: ', image.mean())

threshold_image = (image>70).astype('int')
plt.imshow(threshold_image, cmap="gray")
print('Shape: ', image.shape)
print('Max pixel value: ', image.max())
print('Min pixel value: ', image.min())

sample_mask = df_masks['image_id'][2]
path_mask = '../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + sample_mask

# read the mask using skimage
mask = imread(path_mask)
plt.imshow(mask, cmap='gray')
print('Shape: ', mask.shape)
print('Max pixel value: ', mask.max())
print('Min pixel value: ', mask.min())



# Get lists of images and their masks.
image_id_list = list(df_masks['image_id'])
mask_id_list = list(df_masks['mask_id'])
test_id_list = list(df_test['image_id'])

# Create empty arrays

X_train = np.zeros((len(image_id_list), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(image_id_list), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

X_test = np.zeros((NUM_TEST_IMAGES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

# X_train




for i, image_id in enumerate(image_id_list):
    path_image = '../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_images/BBBC005_v1_images/' + image_id

    # read the image using skimage
    image = imread(path_image)

    # Apply transformations
    # Resize the image
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    #     # Rotate the image by a random angle between -20 and 20 degrees
    #     angle = np.random.uniform(low=-20, high=20)
    #     image = rotate(image, angle, mode='constant', preserve_range=True)

    #     # Flip the image horizontally with 50% probability
    #     if np.random.rand() < 0.5:
    #         image = np.fliplr(image)

    #     # Flip the image vertically with 50% probability
    #     if np.random.rand() < 0.5:
    #         image = np.flipud(image)

    # Use np.expand_dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
    image = np.expand_dims(image, axis=-1)

    # Insert the image into X_train
    X_train[i] = image

print('X-Train_Shape:', X_train.shape)

# Y_train


for i, mask_id in enumerate(mask_id_list):
    path_mask = '../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id

    # read the image using skimage
    mask = imread(path_mask)

    # resize the image
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    #     # Rotate the image by a random angle between -20 and 20 degrees
    #     angle = np.random.uniform(low=-20, high=20)
    #     mask = rotate(mask, angle, mode='constant', preserve_range=True)

    #     # Flip the image horizontally with 50% probability
    #     if np.random.rand() < 0.5:
    #         mask = np.fliplr(mask)

    #     # Flip the image vertically with 50% probability
    #     if np.random.rand() < 0.5:
    #         mask = np.flipud(mask)

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
    mask = np.expand_dims(mask, axis=-1)

    # insert the image into Y_Train
    Y_train[i] = mask

Y_train.shape

print('Y-Train_Shape:', Y_train.shape)

for i, image_id in enumerate(test_id_list):
    path_image = '../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_images/BBBC005_v1_images/' + image_id

    # read the image using skimage
    image = imread(path_image)

    # resize the image
    image = resize(image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    #      # Rotate the image by a random angle between -20 and 20 degrees
    #     angle = np.random.uniform(low=-20, high=20)
    #     image = rotate(image, angle, mode='constant', preserve_range=True)

    #     # Flip the image horizontally with 50% probability
    #     if np.random.rand() < 0.5:
    #         image = np.fliplr(image)

    #     # Flip the image vertically with 50% probability
    #     if np.random.rand() < 0.5:
    #         image = np.flipud(image)

    # use np.expand dims to add a channel axis so the shape becomes (IMG_HEIGHT, IMG_WIDTH, 1)
    image = np.expand_dims(image, axis=-1)

    # insert the image into X_test
    X_test[i] = image

X_test.shape

print('X-Test_Shape:', X_test.shape)

inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

# #Training model
# filepath = "model_seg.keras"

# earlystopper = EarlyStopping(patience=10, verbose=1)

# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
#                              save_best_only=True, mode='min')

# callbacks_list = [earlystopper, checkpoint]

# losses = []  # Initialize empty list to store losses

# for epoch in range(25):
#     print(f"Epoch {epoch+1}/{25}")
#     history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=1,
#                         callbacks=callbacks_list)

#     # Append the training loss to the list
#     losses.append(history.history['loss'][0])

#     print("Training loss:", losses[-1])


model.load_weights('model_seg.keras')

test_preds = model.predict(X_test)

preds_test_thresh = (test_preds >= 0.5).astype(np.uint8)




plt.figure(figsize=(10,10))
plt.axis('Off')

# Our subplot will contain 3 rows and 3 columns
# plt.subplot(nrows, ncols, plot_number)


# == row 1 ==

# image
plt.subplot(3,3,1)
test_image = X_test[1, :, :, 0]
plt.imshow(test_image)
plt.title('Test Image', fontsize=14)
plt.axis('off')


# true mask
plt.subplot(3,3,2)
mask_id = df_test.loc[1,'mask_id']
path_mask = '../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id
mask = imread(path_mask)
mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
plt.imshow(mask, cmap='gray')
plt.title('True Mask', fontsize=14)
plt.axis('off')

# predicted mask
plt.subplot(3,3,3)
test_mask = preds_test_thresh[1, :, :, 0]
plt.imshow(test_mask, cmap='gray')
plt.title('Pred Mask', fontsize=14)
plt.axis('off')

# == row 2 ==

# image
plt.subplot(3,3,4)
test_image = X_test[2, :, :, 0]
plt.imshow(test_image)
plt.title('Test Image', fontsize=14)
plt.axis('off')


# true mask
plt.subplot(3,3,5)
mask_id = df_test.loc[2,'mask_id']
path_mask = '../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id
mask = imread(path_mask)
mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
plt.imshow(mask, cmap='gray')
plt.title('True Mask', fontsize=14)
plt.axis('off')

# predicted mask
plt.subplot(3,3,6)
test_mask = preds_test_thresh[2, :, :, 0]
plt.imshow(test_mask, cmap='gray')
plt.title('Pred Mask', fontsize=14)
plt.axis('off')

# == row 3 ==

# image
plt.subplot(3,3,7)
test_image = X_test[3, :, :, 0]
plt.imshow(test_image)
plt.title('Test Image', fontsize=14)
plt.axis('off')


# true mask
plt.subplot(3,3,8)
mask_id = df_test.loc[3,'mask_id']
path_mask = '../input/synthetic-cell-images-and-masks-bbbc005-v1/bbbc005_v1_ground_truth/BBBC005_v1_ground_truth/' + mask_id
mask = imread(path_mask)
mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
plt.imshow(mask, cmap='gray')
plt.title('True Mask', fontsize=14)
plt.axis('off')

# predicted mask
plt.subplot(3,3,9)
test_mask = preds_test_thresh[3, :, :, 0]
plt.imshow(test_mask, cmap='gray')
plt.title('Pred Mask', fontsize=14)
plt.axis('off')


plt.tight_layout()
plt.show()

# Load the best weights
model.load_weights('model_seg.keras')

# Make predictions
test_preds = model.predict(X_test)

# Threshold the predictions
preds_test_thresh = (test_preds >= 0.5).astype(np.uint8)

# Calculate IoU, precision, and recall
iou_scores = []
precision_scores = []
recall_scores = []

for i in range(len(X_test)):
    y_true = X_test[i].flatten()  # Flatten the ground truth mask
    y_pred = preds_test_thresh[i].flatten()  # Flatten the predicted mask

    # Calculate IoU
    intersection = np.sum(y_true * y_pred)
    union = np.sum((y_true + y_pred) > 0)
    iou = intersection / union
    iou_scores.append(iou)

    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    precision_scores.append(precision)
    recall_scores.append(recall)

# Compute average IoU, precision, and recall
avg_iou = np.mean(iou_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)

print("Average IoU:", avg_iou)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)


# losses = [0.09075859189033508, 0.03579949215054512, 0.03036349266767502, 0.027521677315235138, 0.025623491033911705, 0.02418792061507702, 0.023153889924287796, 0.022085288539528847, 0.021677324548363686, 0.020515913143754005, 0.01991613209247589, 0.019276088103652, 0.01883748359978199, 0.01824432983994484, 0.017943434417247772, 0.017162058502435684, 0.016766401007771492, 0.016265228390693665, 0.01595088467001915, 0.015394375659525394, 0.014993191696703434, 0.014629657380282879, 0.01428668387234211, 0.01395194511860609, 0.013515535742044449]
#
# # Generate x-axis values (epochs)
# epochs = range(1, len(losses) + 1)
#
# # Plotting the graph
# plt.plot(epochs, losses, label='Training loss')
# plt.title('Loss-Augmentations(Rotation, Flipping and Color Jitter)')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

