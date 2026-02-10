# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 19:58:59 2023

@author: SYED ZAHEER HOSSAIN
"""

# Importing necessary libraries
from keras_facenet import FaceNet
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import time
import cv2

# Initialize the FaceNet embedder
embedder = FaceNet()

# Path to the "preprocessed" folder
# base_dir = 'E:\DATA PREPROCESS/preprocessed'
# base_dir = "E:\MACHINE LEARNING\dataraw100"
base_dir = "E:/dataset all combined/comb-preprocessed"
# base_dir = "E:\MACHINE LEARNING\output2"
output_dir = "E:\MACHINE LEARNING\CSV_Final_233"
original = "original"

#code for calculation of time

start = time.time()
timetot=[]
timet = ["No. of images" , "Time Taken"]
timetot.append(timet)

counter = add = 30
j=0


# Initialize lists to store face embeddings and corresponding labels for training and testing
train_embeddings = []
train_labels = []
test_embeddings = []
test_labels = []

z=1
no = 30
# crop = "cropped"
crop = "cropped"

# Create the csv output folder if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

subfolders = os.listdir(base_dir)
for subdir in subfolders:
    if(z):
        z+=1
        subdir_path = os.path.join(base_dir, subdir, original)
        if os.path.isdir(subdir_path):  # Check if it's a directory
            image_paths = [os.path.join(subdir_path, filename) for filename in os.listdir(subdir_path)]
            
            # split image paths into training and testing datasets with a 50-50 ratio
            train_paths, test_paths = train_test_split(image_paths, test_size=0.5, random_state=42)
            
            # Initialize embedding lists for training and testing
            train_embeddings_subfolder = []
            test_embeddings_subfolder = []
            
            # Embed images for training
            for image_path in train_paths:
                # image = Image.open(image_path)
                
                # Convert PIL Image to NumPy array
                # image_array = np.array(image)
                image_array = cv2.imread(image_path)
                image_array = cv2.resize(image_array,(100,100))
                
                # Embed the image and append to training embeddings
                embedding = embedder.embeddings([image_array])[0]
                image_size = image_array.shape
                train_embeddings_subfolder.append(embedding)
                train_labels.append(subdir)
                
                print(j)
                j=j+1
                
                if(j == counter):
                    timet=[]
                    end = time.time()
                    tott = end - start
                    timet.append(counter)
                    timet.append(tott)
                    timetot.append(timet)
                    counter += add
                    print(tott)
            
            # Embed images for testing
            for image_path in test_paths:
                # image = Image.open(image_path)
                
                # Convert PIL Image to NumPy array
                # image_array = np.array(image)
                image_array = cv2.imread(image_path)
                image_array = cv2.resize(image_array,(100,100))
                
                # Embed the image and append to testing embeddings
                embedding = embedder.embeddings([image_array])[0]
                test_embeddings_subfolder.append(embedding)
                test_labels.append(subdir)
                
                print(j)
                j=j+1
                
                if(j == counter):
                    timet=[]
                    end = time.time()
                    tott = end - start
                    timet.append(counter)
                    timet.append(tott)
                    timetot.append(timet)
                    counter += add
                    print(tott)
        
        # Append subfolder embeddings to the main training and testing lists
        train_embeddings.extend(train_embeddings_subfolder)
        test_embeddings.extend(test_embeddings_subfolder)

# Convert embeddings and labels to DataFrames for training and testing
train_embeddings_df = pd.DataFrame(train_embeddings)
train_labels_df = pd.DataFrame(train_labels, columns=['label'])
test_embeddings_df = pd.DataFrame(test_embeddings)
test_labels_df = pd.DataFrame(test_labels, columns=['label'])

# Concatenate embeddings and labels DataFrames for training and testing
train_result_df = pd.concat([train_embeddings_df, train_labels_df], axis=1)
test_result_df = pd.concat([test_embeddings_df, test_labels_df], axis=1)

# Save embeddings and labels of the training dataset to a CSV file
output_train_csv_path = os.path.join(output_dir,f"facenet_train_embeddings_{crop}_{z-1}.csv")
train_result_df.to_csv(output_train_csv_path, index=False)
print(f"Training embeddings saved to {output_train_csv_path}")

# Save embeddings and labels of the testing dataset to a CSV file
output_test_csv_path = os.path.join(output_dir,f"facenet_test_embeddings_{crop}_{z-1}.csv")
test_result_df.to_csv(output_test_csv_path, index=False)
print(f"Testing embeddings saved to {output_test_csv_path}")

# Load training embeddings and labels from CSV
train_embeddings_df = pd.read_csv(output_train_csv_path)
trainX = train_embeddings_df.drop(columns=['label']).values
trainy = train_embeddings_df['label'].values

# Load testing embeddings and labels from CSV
test_embeddings_df = pd.read_csv(output_test_csv_path)
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


#code for calculation of time

end = time.time()
tott = end - start

#code for calculation of time

end = time.time()
tott = end - start

avg = (tott / (no*12))

print(f"Average time for image processing is {avg}")
print(f"size of dataset image is {image_size}")

# timet=[]
# timet.append(j)
# timet.append(tott)
# timetot.append(timet)
 
# pd.DataFrame(timetot).to_csv("E:\MACHINE LEARNING\IMAGE PROCESSING/facetime.csv")