# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:03:18 2024

@author: SYED ZAHEER HOSSAIN
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from keras.applications.vgg19 import VGG19  # Import VGG19 model
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

# Load pre-trained VGG19 model
vgg_model = VGG19(weights='imagenet', include_top=False, pooling='avg')

# Define a function to extract features using the VGG19 model
def extract_vgg_features(image_path):
    img = Image.open(image_path)
    img = img.resize((100, 100))  # Resize image to the input size of VGG16
    img = img.convert('RGB')
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    # Preprocess input according to the requirements of VGG19 model
    x = preprocess_input(x)
    features = vgg_model.predict(x)
    return features.flatten()

# Path to the input master folder
input_master_folder = r'E:\MACHINE LEARNING/outputfinal233'



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
        
        # Path to training and testing subfolders within the numerical subfolder
        train_folder = os.path.join(subdir_path, 'training')
        test_folder = os.path.join(subdir_path, 'testing')
        
        # Extract features for training
        for image_file in os.listdir(train_folder):
            image_path = os.path.join(train_folder, image_file)
            features = extract_vgg_features(image_path)
            train_data.append([features.tolist(), subdir])
        
        # Extract features for testing
        for image_file in os.listdir(test_folder):
            image_path = os.path.join(test_folder, image_file)
            features = extract_vgg_features(image_path)
            test_data.append([features.tolist(), subdir])

# Convert lists to DataFrames
train_df = pd.DataFrame(train_data, columns=['feature_Vector', 'Label'])
test_df = pd.DataFrame(test_data, columns=['feature_Vector', 'Label'])

# Write DataFrames to CSV
train_df.to_csv('E:\MACHINE LEARNING/csv_antispoof233/train_embeddings_vgg19_cropped.csv', index=False)
test_df.to_csv('E:\MACHINE LEARNING/csv_antispoof233/test_embeddings_vgg19_cropped.csv', index=False)

print("Data saved successfully to 'E:\MACHINE LEARNING/csv_antispoof233/train_embeddings_vgg19_cropped.csv'")