# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:58:35 2024

@author: SYED ZAHEER HOSSAIN
"""

import os
import shutil
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage import exposure
import csv
import cv2

# Path to the input master folder
input_master_folder = r'E:\MACHINE LEARNING\output2'

# Path to the output master folder
output_master_folder = r'E:\MACHINE LEARNING\outputfinal'

# path to csv saving folder
csv_folder = r'E:\MACHINE LEARNING\csv_antispoof'

# Create the csv output folder if it doesn't exist
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# Create the output master folder if it doesn't exist
if not os.path.exists(output_master_folder):
    os.makedirs(output_master_folder)

# Function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# Function to determine classification based on the image name
def determine_classification(image_name):
    if 'fake' in image_name.lower():
        return 0  # Fake
    else:
        return 1  # Original

# Initialize lists to store features, labels, and classifications for training and testing
train_data = []
test_data = []

# Iterate through each numerical subfolder in the input master folder
for subdir in os.listdir(input_master_folder):
    print(subdir)
    subdir_path = os.path.join(input_master_folder, subdir)
    if os.path.isdir(subdir_path):  # Check if it's a directory
        # Create a corresponding numerical subfolder in the output master folder
        output_subdir_path = os.path.join(output_master_folder, subdir)
        os.makedirs(output_subdir_path, exist_ok=True)
        
        # Create training and testing subfolders within the numerical subfolder
        train_folder = os.path.join(output_subdir_path, 'training')
        test_folder = os.path.join(output_subdir_path, 'testing')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        # Copy and rename the first 6 images from "Fake" and "Original" folders into the training subfolder
        for image_type in ['Fake', 'Original']:
            for i in range(1, 7):
                src_files = [file for file in os.listdir(os.path.join(subdir_path, image_type)) if is_image_file(file)]
                if src_files:
                    src_path = os.path.join(subdir_path, image_type, src_files[i-1])
                    dst_path = os.path.join(train_folder, f'{image_type.lower()}{i}.jpg')
                    shutil.copy(src_path, dst_path)
        
        # Copy and rename the last 6 images from "Fake" and "Original" folders into the testing subfolder
        for image_type in ['Fake', 'Original']:
            for i in range(7, 13):
                src_files = [file for file in os.listdir(os.path.join(subdir_path, image_type)) if is_image_file(file)]
                if src_files:
                    src_path = os.path.join(subdir_path, image_type, src_files[i-1])
                    dst_path = os.path.join(test_folder, f'{image_type.lower()}{i}.jpg')
                    shutil.copy(src_path, dst_path)
        
        # Extract HOG features for training
        for image_path in os.listdir(train_folder):
            # image = Image.open(os.path.join(train_folder, image_path)).convert('L')  # Convert to grayscale
            # image_array = np.array(image)
            image_array = cv2.imread(os.path.join(train_folder, image_path))
            image_array= cv2.resize(image_array, (160,160))
            image_array = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
            # Extract HOG features
            fd, hog_image = hog(image_array, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            # Rescale histogram for better visualization
            # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            classification = determine_classification(image_path)
            train_data.append([fd.tolist(), subdir, classification])
        
        # Extract HOG features for testing
        for image_path in os.listdir(test_folder):
            # image = Image.open(os.path.join(test_folder, image_path)).convert('L')  # Convert to grayscale
            # image_array = np.array(image)
            image_array = cv2.imread(os.path.join(test_folder, image_path))
            image_array= cv2.resize(image_array, (160,160))
            image_array = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
            # Extract HOG features
            fd, hog_image = hog(image_array, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            print(fd)
            # Rescale histogram for better visualization
            # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            classification = determine_classification(image_path)
            test_data.append([fd.tolist(), subdir, classification])


# Write training data to CSV
with open(os.path.join(csv_folder, 'train_embeddings.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['feature_Vector', 'Label', 'Classification'])
    writer.writerows(train_data)

# Write testing data to CSV
with open(os.path.join(csv_folder, 'test_embeddings.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['feature_Vector', 'Label', 'Classification'])
    writer.writerows(test_data)