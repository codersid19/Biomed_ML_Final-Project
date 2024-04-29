import os
import matplotlib.pyplot as plt
from cellpose import models
import numpy as np
import cv2


def perform_instance_segmentation(image_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize Cellpose model
    model = models.Cellpose(gpu=False, model_type='cyto')  # Use 'cyto' model for cell/nuclear segmentation

    # Counter for mask filenames
    mask_counter = 1

    # Process each image in the input folder
    for filename in os.listdir(image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Load the image
            img_path = os.path.join(image_folder, filename)
            img = plt.imread(img_path)

            # Perform instance segmentation
            masks, _, _, _ = model.eval([img], diameter=30)  # Pass the image in a list

            # Combine masks into a single image
            img_result = np.zeros_like(img)
            for mask in masks:
                img_result += mask[:, :, None]

            # Save the segmentation mask
            mask_filename = f"img_{mask_counter}_mask.png"
            mask_path = os.path.join(output_folder, mask_filename)
            cv2.imwrite(mask_path, img_result)

            print("Mask saved:", mask_path)

            # Increment mask counter
            mask_counter += 1


# Example usage
if __name__ == "__main__":
    image_folder = 'S:/College Folder/UCF/Biomed/Final Project/Your Source Code/Code/data'  # Replace with the path to your input image folder
    output_folder = 'S:/College Folder/UCF/Biomed/Final Project/Your Source Code/Code/masks'  # Replace with the path to your output mask folder
    perform_instance_segmentation(image_folder, output_folder)
