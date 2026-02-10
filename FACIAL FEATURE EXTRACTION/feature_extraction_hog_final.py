# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:38:31 2024

@author: SYED ZAHEER HOSSAIN
"""

import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn import preprocessing

# Path to the "preprocessed" folder
# base_dir = "E:\DATA PREPROCESS\preprocessed"
base_dir = "E:\DataSet"
# base_dir = "E:\PREPROCESSED-CLEANED-ALL"

output_dir = "E:\MACHINE LEARNING\CSV_Final"
# base_dir = input("enter the address of folder where datas are stored")

# Initialize lists to store face hog_features and corresponding labels for training and testing
train_hog_features = []
train_labels = []
test_hog_features = []
test_labels = []

# radius=2
# numpoints=16
# size1=size2=160

z=1

original = "original"

for subdir in os.listdir(base_dir):
    if(os.path.exists(os.path.join(base_dir,subdir,"Original"))):
       os.rename(os.path.join(base_dir,subdir,"Original"), os.path.join(base_dir,subdir,"original"))

    if(os.path.exists(os.path.join(base_dir,subdir,"Fake"))):
       os.rename(os.path.join(base_dir,subdir,"Fake"), os.path.join(base_dir,subdir,"fake"))    
    # for i in range(30):
    if(z):
        z+=1
        print(subdir)
        subdir_path = os.path.join(base_dir, subdir,original)
        if os.path.isdir(subdir_path):  # Check if it's a directory
            image_paths = [os.path.join(subdir_path, filename) for filename in os.listdir(subdir_path)]
            
            # split image paths into training and testing datasets with a 50-50 ratio
            train_paths, test_paths = train_test_split(image_paths, test_size=0.5, random_state=42)
            
            # Initialize embedding lists for training and testing
            train_hog_features_subfolder = []
            test_hog_features_subfolder = []
            
            # Embed images for training
            for image_path in train_paths:
                image_array = cv2.imread(image_path)
                image_array= cv2.resize(image_array, (200,200))
                image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
               
                # Embed the image and append to training hog_features
                fd, _ = hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            
                train_hog_features_subfolder.append(fd)
                train_labels.append(subdir)
            
            # Embed images for testing
            for image_path in test_paths:
                image_array = cv2.imread(image_path)
                image_array= cv2.resize(image_array, (200,200))
                image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
                
                # Embed the image and append to training hog_features
                fd, _ = hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
                
                test_hog_features_subfolder.append(fd)
        
            # Append subfolder hog_features to the main training and testing lists
            train_hog_features.extend(train_hog_features_subfolder)
            test_hog_features.extend(test_hog_features_subfolder)
        
# Convert hog_features and labels to DataFrames for training and testing
train_hog_features_df = pd.DataFrame(train_hog_features)
train_labels_df = pd.DataFrame(train_labels)
test_hog_features_df = pd.DataFrame(test_hog_features)
test_labels_df = pd.DataFrame(test_labels)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save hog_features and labels of the training dataset to a CSV file
output_train_csv_path = os.path.join(output_dir,f"train_hog_features_{z-1}.csv")
train_hog_features_df.to_csv(output_train_csv_path, index=False)
print(f"Training hog_features saved to {output_train_csv_path}")

# Save hog_features and labels of the testing dataset to a CSV file
output_test_csv_path = os.path.join(output_dir,f"test_hog_features_{z-1}.csv")
test_hog_features_df.to_csv(output_test_csv_path, index=False)
print(f"Testing hog_features saved to {output_test_csv_path}")

# Save hog_features and labels of the training dataset to a CSV file
output_train_csv_path = os.path.join(output_dir,f"train_hog_labels_{z-1}.csv")
train_labels_df.to_csv(output_train_csv_path, index=False)
print(f"Training hog_features saved to {output_train_csv_path}")

# Save hog_features and labels of the training dataset to a CSV file
output_test_csv_path = os.path.join(output_dir,f"test_hog_labels_{z-1}.csv")
test_labels_df.to_csv(output_test_csv_path, index=False)
print(f"Training hog_features saved to {output_train_csv_path}")