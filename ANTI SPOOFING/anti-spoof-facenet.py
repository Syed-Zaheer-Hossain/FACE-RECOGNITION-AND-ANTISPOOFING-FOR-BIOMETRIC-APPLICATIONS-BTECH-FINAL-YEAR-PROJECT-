# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:09:30 2024

@author: SYED ZAHEER HOSSAIN
"""

z=1

import os
import shutil
import numpy as np
from PIL import Image
from keras_facenet import FaceNet
import csv
import cv2

# Initialize the FaceNet embedder
embedder = FaceNet()

# Path to the input master folder
input_master_folder = r'E:/dataset all combined/comb-preprocessed'
# input_master_folder = r'E:\DataSet'
# input_master_folder = r'E:\PROJECT DATA 2024 FINAL'


# Path to the output master folder
output_master_folder = r'E:\MACHINE LEARNING\outputfinal233'

# path to csv saving folder
csv_folder = r'E:\MACHINE LEARNING\csv_antispoof233'

# Create the output master folder if it doesn't exist
if not os.path.exists(output_master_folder):
    os.makedirs(output_master_folder)
    
# Create the csv output folder if it doesn't exist
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# Function to check if a file is an image
def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

# Function to determine classification based on the image name
def determine_classification(image_name):
    if 'fake' in image_name.lower():
        return 0  # Fake
    else:
        return 1  # Original

# Initialize lists to store embeddings, labels, and classifications for training and testing
train_data = []
test_data = []

# Iterate through each numerical subfolder in the input master folder
for subdir in os.listdir(input_master_folder):
    print(subdir)
    if(z):
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
                        image = cv2.imread(src_path)
                        dst_path = os.path.join(train_folder, f'{image_type.lower()}{i}.png')
                        image_array = cv2.resize(image, (100,100))
                        cv2.imwrite(dst_path, image_array)
                        # shutil.copy(src_path, dst_path)
            
            # Copy and rename the last 6 images from "Fake" and "Original" folders into the testing subfolder
            for image_type in ['Fake', 'Original']:
                for i in range(7, 13):
                    src_files = [file for file in os.listdir(os.path.join(subdir_path, image_type)) if is_image_file(file)]
                    if src_files:
                        src_path = os.path.join(subdir_path, image_type, src_files[i-1])
                        image = cv2.imread(src_path)
                        dst_path = os.path.join(test_folder, f'{image_type.lower()}{i}.png')
                        image_array = cv2.resize(image, (100,100))
                        cv2.imwrite(dst_path, image_array)
                        # shutil.copy(src_path, dst_path)
            
            # Embed images for training
            for image_path in os.listdir(train_folder):
                # image = Image.open(os.path.join(train_folder, image_path))
                # image_array = np.array(image)
                image_array = cv2.imread(os.path.join(train_folder, image_path))
                # image_array = cv2.resize(image_array, (160,160))
                # image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
                embedding = embedder.embeddings([image_array])[0]
                print(embedding)
                classification = determine_classification(image_path)
                train_data.append([embedding.tolist(), subdir, classification])
            
            # Embed images for testing
            for image_path in os.listdir(test_folder):
                # image = Image.open(os.path.join(test_folder, image_path))
                # image_array = np.array(image)
                image_array = cv2.imread(os.path.join(test_folder, image_path))
                # image_array = cv2.resize(image_array, (160,160))
                # image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
                embedding = embedder.embeddings([image_array])[0]
                classification = determine_classification(image_path)
                test_data.append([embedding.tolist(), subdir, classification])
    z+=1

# Write training data to CSV
with open(os.path.join(csv_folder,'train_embeddings_cropped_facenet.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['feature_Vector', 'Label', 'Classification'])
    writer.writerows(train_data)

# Write testing data to CSV
with open(os.path.join(csv_folder,'test_embeddings_cropped_facenet.csv'), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['feature_Vector', 'Label', 'Classification'])
    writer.writerows(test_data)