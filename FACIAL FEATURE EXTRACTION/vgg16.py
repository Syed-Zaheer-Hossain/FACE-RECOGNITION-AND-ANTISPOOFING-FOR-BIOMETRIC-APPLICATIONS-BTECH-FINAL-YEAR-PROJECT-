# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:18:14 2024

@author: SYED ZAHEER HOSSAIN
"""

# Importing necessary libraries
from keras.applications.vgg16 import VGG16  # Import VGG16 model
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import time

# Load pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg',input_shape=(100, 100, 3))

# Define a function to extract features using the VGG16 model
def extract_vgg_features(image_path):
    img = Image.open(image_path)
    img = img.resize((100, 100))  # Resize image to the input size of VGG16
    img = img.convert('RGB')
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    features = vgg_model.predict(x)
    return features.flatten()

# Path to the main directory containing numerical-named folders
base_dir = r'E:\dataset all combined\comb-preprocessed'
# base_dir = r'E:\MACHINE LEARNING\output2'

csv = "E:\MACHINE LEARNING\CSV_Final_233"

if not os.path.exists(csv):
    os.makedirs(csv)

# Initialize lists to store face embeddings and corresponding labels for training and testing
train_embeddings = []
train_labels = []
test_embeddings = []
test_labels = []

z=1
no=233
crop = "cropped"
# crop = "uncropped"

start = time.time()

# Iterate over numerical-named folders
for folder_name in os.listdir(base_dir):
    if(z<=no):
        z+=1
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            # For each numerical-named folder, process only the "original" subfolder
            original_folder_path = os.path.join(folder_path, 'original')
            if os.path.isdir(original_folder_path):
                image_paths = [os.path.join(original_folder_path, filename) for filename in os.listdir(original_folder_path)]
                
                # split image paths into training and testing datasets with a 50-50 ratio
                train_paths, test_paths = train_test_split(image_paths, test_size=0.5, random_state=42)
                
                # Embed images for training
                for image_path in train_paths:
                    embedding = extract_vgg_features(image_path)
                    train_embeddings.append(embedding)
                    train_labels.append(folder_name)
                
                # Embed images for testing
                for image_path in test_paths:
                    embedding = extract_vgg_features(image_path)
                    test_embeddings.append(embedding)
                    test_labels.append(folder_name)

# Convert embeddings and labels to DataFrames for training and testing
train_embeddings_df = pd.DataFrame(train_embeddings)
train_labels_df = pd.DataFrame(train_labels, columns=['label'])
test_embeddings_df = pd.DataFrame(test_embeddings)
test_labels_df = pd.DataFrame(test_labels, columns=['label'])

# Save embeddings and labels of the training dataset to CSV files
output_train_embeddings_csv_path = os.path.join(csv,f'train_embeddings_{crop}_{no}.csv')
output_train_labels_csv_path = os.path.join(csv,f'train_labels_{crop}_{no}.csv')
output_test_embeddings_csv_path = os.path.join(csv,f'test_embeddings_{crop}_{no}.csv')
output_test_labels_csv_path = os.path.join(csv,f'test_labels_{crop}_{no}.csv')

train_embeddings_df.to_csv(output_train_embeddings_csv_path, index=False)
train_labels_df.to_csv(output_train_labels_csv_path, index=False)
test_embeddings_df.to_csv(output_test_embeddings_csv_path, index=False)
test_labels_df.to_csv(output_test_labels_csv_path, index=False)

print(f"Training embeddings saved to {output_train_embeddings_csv_path}")
print(f"Training labels saved to {output_train_labels_csv_path}")
print(f"Testing embeddings saved to {output_test_embeddings_csv_path}")
print(f"Testing labels saved to {output_test_labels_csv_path}")

# Concatenate embeddings and labels DataFrames for training and testing
train_result_df = pd.concat([train_embeddings_df, train_labels_df], axis=1)
test_result_df = pd.concat([test_embeddings_df, test_labels_df], axis=1)

# Save embeddings and labels of the training dataset to a CSV file
output_train_csv_path = os.path.join(csv,f'train_embeddings_and_labels_vgg16_{crop}_{no}.csv')
output_test_csv_path = os.path.join(csv,f'test_embeddings_and_labels_vgg16_{crop}_{no}.csv')

train_result_df.to_csv(output_train_csv_path, index=False)
test_result_df.to_csv(output_test_csv_path, index=False)

print(f"Training embeddings and labels saved to {output_train_csv_path}")
print(f"Testing embeddings and labels saved to {output_test_csv_path}")



import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

# Load training embeddings and labels from CSV
train_embeddings_df = pd.read_csv(os.path.join(csv,f'train_embeddings_and_labels_vgg16_{crop}_{no}.csv'))  # Adjust path
trainX = train_embeddings_df.drop(columns=['label']).values
trainy = train_embeddings_df['label'].values

# Load testing embeddings and labels from CSV
test_embeddings_df = pd.read_csv(os.path.join(csv,f'test_embeddings_and_labels_vgg16_{crop}_{no}.csv') ) # Adjust path
testX = test_embeddings_df.drop(columns=['label']).values
testy = test_embeddings_df['label'].values

# Normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# Label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# SVM classifier with a radial basis function kernel
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(trainX, trainy)
yhat_train_svm = svm_model.predict(trainX)
yhat_test_svm = svm_model.predict(testX)
accuracy_train_svm = accuracy_score(trainy, yhat_train_svm)
accuracy_test_svm = accuracy_score(testy, yhat_test_svm)

print('SVM Accuracy: train=%.3f, test=%.3f' % (accuracy_train_svm * 100, accuracy_test_svm * 100))




end = time.time()

timet = (end - start)/1824

print(f"time : {timet}s")