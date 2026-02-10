# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:10:20 2024

@author: SYED ZAHEER HOSSAIN
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import csv
import shutil
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet201
from sklearn.preprocessing import MinMaxScaler

# Function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# Define function to extract features using DenseNet201 model
def extract_densenet_features(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize image to match DenseNet201 input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    features = densenet_model.predict(img_array)
    return features.flatten()  # Remove .tolist() for better normalization

# Define function to determine classification based on image name
def determine_classification(image_name):
    if 'fake' in image_name.lower():
        return 0  # Fake
    else:
        return 1  # Original

# Load pre-trained DenseNet201 model
densenet_model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')

# Path to the input master folder
input_master_folder = r'E:\dataset all combined\comb'

# Path to the output master folder
output_master_folder = r'E:\MACHINE LEARNING\teesty'

# Path to CSV saving folder
csv_folder = r'E:\MACHINE LEARNING\densenetcsv'

# Create the csv output folder if it doesn't exist
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# Initialize lists to store features and labels for training and testing
train_data = []
test_data = []

# Iterate through each numerical subfolder in the input master folder
folders_to_process = 80
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
            features = extract_densenet_features(image_path)
            classification = determine_classification(image_path)
            train_data.append([features, subdir, classification])  # Append as list of features

        # Extract features for testing
        for image_path in os.listdir(test_folder):
            image_path = os.path.join(test_folder, image_path)
            features = extract_densenet_features(image_path)
            classification = determine_classification(image_path)
            test_data.append([features, subdir, classification])  # Append as list of features

# Normalize features
scaler = MinMaxScaler()
train_data_features = np.array([row[0] for row in train_data])
test_data_features = np.array([row[0] for row in test_data])
train_data_features_normalized = scaler.fit_transform(train_data_features)
test_data_features_normalized = scaler.transform(test_data_features)

# Update train_data and test_data with normalized features
for i in range(len(train_data)):
    train_data[i][0] = train_data_features_normalized[i].tolist()
for i in range(len(test_data)):
    test_data[i][0] = test_data_features_normalized[i].tolist()

# Write training data to CSV with rounded feature vectors
with open(os.path.join(csv_folder,'train_embeddings_cropped_densenet.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['feature_Vector', 'Label', 'Classification'])
    for row in train_data:
        rounded_features = [round(feature, 6) for feature in row[0]]  # Round each feature to 10^-6 precision
        writer.writerow([rounded_features] + row[1:])

# Write testing data to CSV with rounded feature vectors
with open(os.path.join(csv_folder,'test_embeddings_cropped_densenet.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['feature_Vector', 'Label', 'Classification'])
    for row in test_data:
        rounded_features = [round(feature, 6) for feature in row[0]]  # Round each feature to 10^-6 precision
        writer.writerow([rounded_features] + row[1:])