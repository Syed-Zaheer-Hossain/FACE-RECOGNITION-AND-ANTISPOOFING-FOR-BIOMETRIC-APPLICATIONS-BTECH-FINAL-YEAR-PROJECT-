# -*- coding: utf-8 -*-
"""
Created on Sun May  5 20:48:42 2024

@author: SYED ZAHEER HOSSAIN
"""

import os
import pandas as pd
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16  # Import VGG16 model
import shutil
import csv

# Load pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Define a function to extract features using the VGG16 model
def extract_vgg_features(image_path):
    # img = Image.open(image_path)
    # x = np.array(img)
    x = cv2.imread(image_path)
    x = cv2.resize(x, (224,224))
    x = np.expand_dims(x, axis=0)
    features = vgg_model.predict(x)
    return features.flatten()

# Function to determine classification based on the image name
def determine_classification(image_name):
    if 'fake' in image_name.lower():
        return 0  # Fake
    else:
        return 1  # Original

# Path to the input master folder
input_master_folder = r'E:/dataset all combined/comb-preprocessed'
input_master_folder = r'E:\dataset all combined\comb'

# Path to the output master folder
output_master_folder = r'E:\MACHINE LEARNING\outputfinal233'

# path to csv saving folder
csv_folder = r'E:\MACHINE LEARNING\csv_antispoof233'

# Create the csv output folder if it doesn't exist
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# Create the output master folder if it doesn't exist
if not os.path.exists(output_master_folder):
    os.makedirs(output_master_folder)

# Function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# Initialize lists to store features and labels for training and testing
train_data = []
test_data = []

# Iterate through each numerical subfolder in the input master folder
folders_to_process = 233
processed_folders = 0
for subdir in os.listdir(input_master_folder):
    if processed_folders >= folders_to_process:
        break
    subdir_path = os.path.join(input_master_folder, subdir)
    if os.path.isdir(subdir_path):  # Check if it's a directory
        processed_folders += 1
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
        
        # Extract features for training
        for image_path in os.listdir(train_folder):
            image_path = os.path.join(train_folder, image_path)
            features = extract_vgg_features(image_path)
            classification = determine_classification(image_path)
            train_data.append([features.tolist(), subdir, classification])
        
        # Extract features for testing
        for image_path in os.listdir(test_folder):
            image_path = os.path.join(test_folder, image_path)
            features = extract_vgg_features(image_path)
            classification = determine_classification(image_path)
            test_data.append([features.tolist(), subdir, classification])

# Convert lists to DataFrames
train_df = pd.DataFrame(train_data, columns=['feature_Vector', 'Label', 'Classification'])
test_df = pd.DataFrame(test_data, columns=['feature_Vector', 'Label', 'Classification'])

# Write training data to CSV
with open(os.path.join(csv_folder,'train_embeddings_uncropped_vgg16.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['feature_Vector', 'Label', 'Classification'])
    writer.writerows(train_data)

# Write testing data to CSV
with open(os.path.join(csv_folder,'test_embeddings_uncropped_vgg16.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['feature_Vector', 'Label', 'Classification'])
    writer.writerows(test_data)