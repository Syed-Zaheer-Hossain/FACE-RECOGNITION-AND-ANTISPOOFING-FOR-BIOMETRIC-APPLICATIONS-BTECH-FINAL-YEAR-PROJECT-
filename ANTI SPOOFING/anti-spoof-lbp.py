# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 16:00:23 2024

@author: SYED ZAHEER HOSSAIN
"""


import os
import shutil
import numpy as np
import cv2
from PIL import Image
from keras_facenet import FaceNet
import csv
from skimage.feature import local_binary_pattern
import tensorflow as tf
import time
from sklearn import preprocessing


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    

def lbp_feature_extraction(image,height,width):
    img_lbp = np.zeros((height, width),np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(image, i, j)
            
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])        
    fd=hist_lbp
    # print(np.ravel(hist_lbp))
    fd = np.ravel(fd)
    fd = fd.reshape(1, -1)
    fd=preprocessing.normalize(fd)
    # print(fd.shape)
    return fd



# Path to the input master folder
input_master_folder = r'E:\dataset all combined/comb'
# input_master_folder = r'E:\DataSet'

# Path to the output master folder
output_master_folder = r'E:\MACHINE LEARNING\outputfinaluncropped233'

# path to csv saving folder
# csv_folder = r'E:\MACHINE LEARNING\csv_antispoof233'

radius=2
numpoints=16
size1=size2=200

start = time.time()

# Create the output master folder if it doesn't exist
if not os.path.exists(output_master_folder):
    os.makedirs(output_master_folder)
    
# Create the csv output folder if it doesn't exist
# if not os.path.exists(csv_folder):
    # os.makedirs(csv_folder)

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
    subdir_path = os.path.join(input_master_folder, subdir)
    print(subdir)
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
        
#         # Embed images for training
#         for image_path in os.listdir(train_folder):
#             image = Image.open(os.path.join(train_folder, image_path))
#             image_array = np.array(image)
#             image_array= cv2.resize(image_array, dsize=[size1,size2])
#             image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
#             embedding = lbp_feature_extraction(image_gray, height= len(image_gray), width=len(image_gray[0]))
#             # embedding = tf.reshape(embedding,[size1*size2]).numpy()
#             classification = determine_classification(image_path)
#             train_data.append([embedding[0].tolist(), subdir, classification])
        
#         # Embed images for testing
#         for image_path in os.listdir(test_folder):
#             image = Image.open(os.path.join(test_folder, image_path))
#             image_array = np.array(image)
#             image_array= cv2.resize(image_array, dsize=[size1,size2])
#             image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
#             embedding = lbp_feature_extraction(image_gray, height= len(image_gray), width=len(image_gray[0]))
#             # embedding = tf.reshape(embedding,[size1*size2]).numpy()
#             classification = determine_classification(image_path)
#             test_data.append([embedding[0].tolist(), subdir, classification])

# # Write training data to CSV
# with open(os.path.join(csv_folder,'train_embeddings_cropped_lbp.csv'), 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['feature_Vector', 'Label', 'Classification'])
#     writer.writerows(train_data)

# # Write testing data to CSV
# with open(os.path.join(csv_folder,'test_embeddings_cropped_lbp.csv'), 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['feature_Vector', 'Label', 'Classification'])
#     writer.writerows(test_data)
    
# end = time.time()

# timet = end-start