# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 20:49:58 2023

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
import time
import pandas as pd
import os
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# import mahotas

# Path to the "preprocessed" folder
# base_dir = "E:\DATA PREPROCESS\preprocessed"
base_dir = "E:\dataset all combined/comb-preprocessed"

output_dir = "E:\MACHINE LEARNING\CSV_Final_233"
# base_dir = input("enter the address of folder where datas are stored")

#code for calculation of time

start = time.time()
timetot=[]
timet = ["No. of images" , "Time Taken"]
timetot.append(timet)

counter = add = 30
j=0

# Initialize lists to store face lbp_features and corresponding labels for training and testing
train_lbp_features = []
train_labels = []
test_lbp_features = []
test_labels = []

radius=2
numpoints=16
size1=size2=100

z=1
no=233

original = "original"

for subdir in os.listdir(base_dir):
    if(os.path.exists(os.path.join(base_dir,subdir,"Original"))):
       os.rename(os.path.join(base_dir,subdir,"Original"), os.path.join(base_dir,subdir,"original"))

    if(os.path.exists(os.path.join(base_dir,subdir,"Fake"))):
       os.rename(os.path.join(base_dir,subdir,"Fake"), os.path.join(base_dir,subdir,"fake"))    
    # for i in range(30):
    if(z<=no):
        z+=1
        subdir_path = os.path.join(base_dir, subdir,original)
        if os.path.isdir(subdir_path):  # Check if it's a directory
            image_paths = [os.path.join(subdir_path, filename) for filename in os.listdir(subdir_path)]
            
            # split image paths into training and testing datasets with a 50-50 ratio
            train_paths, test_paths = train_test_split(image_paths, test_size=0.5, random_state=42)
            
            # Initialize embedding lists for training and testing
            train_lbp_features_subfolder = []
            test_lbp_features_subfolder = []
            
            # Embed images for training
            for image_path in train_paths:
                image_array = cv2.imread(image_path)
                # image_array= cv2.resize(image_array, dsize=[size1,size2])
                image_array= cv2.resize(image_array, dsize=[100,100])
                image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
                
                image_size = image_gray.shape
                
                # lbp = mahotas.features.lbp(image_gray,12,9)[0]
                
                # Embed the image and append to training lbp_features
                lbp = local_binary_pattern(image_gray, numpoints, radius)
                
                width = len(lbp)
                height = len(lbp[0])
                size1=width
                size2=height
                
                # size1 = lbp.shape[0]
                # size2 = lbp.shape[1]
                
                # print(lbp.shape)
                lbp = tf.reshape(lbp,[size1*size2]).numpy()
                
                # lbp = lbp[:512]
                # print(lbp)
                # desc=LocalBinaryPatterns(50, 20)    
                # lbp = desc.describe(image_gray)
                train_lbp_features_subfolder.append(lbp)
                train_labels.append(subdir)
                
                #code for calculation of time
            
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
                image_array = cv2.imread(image_path)
                # image_array= cv2.resize(image_array, dsize=[size1,size2])
                image_array= cv2.resize(image_array, dsize=[100,100])
                image_gray = cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
                
                
                # Embed the image and append to training lbp_features
                lbp = local_binary_pattern(image_gray, numpoints, radius)
                # print(lbp.shape)
                
                width = len(lbp)
                height = len(lbp[0])
                size1=width
                size2=height
                
                lbp = tf.reshape(lbp,[size1*size2]).numpy()
                
                # lbp = lbp[:512]
                # desc=LocalBinaryPatterns(50, 20)    
                # lbp = desc.describe(image_gray)
                # lbp = mahotas.features.lbp(image_gray,12,9)[0]
                test_lbp_features_subfolder.append(lbp)
                test_labels.append(subdir)
                
                #code for calculation of time
            
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

        
        # Append subfolder lbp_features to the main training and testing lists
        train_lbp_features.extend(train_lbp_features_subfolder)
        test_lbp_features.extend(test_lbp_features_subfolder)
        
# Convert lbp_features and labels to DataFrames for training and testing
train_lbp_features_df = pd.DataFrame(train_lbp_features)
train_labels_df = pd.DataFrame(train_labels)
test_lbp_features_df = pd.DataFrame(test_lbp_features)
test_labels_df = pd.DataFrame(test_labels)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save lbp_features and labels of the training dataset to a CSV file
output_train_csv_path = os.path.join(output_dir,f"train_lbp_cropped_features_{z-1}.csv")
train_lbp_features_df.to_csv(output_train_csv_path, index=False)
print(f"Training lbp_features saved to {output_train_csv_path}")

# Save lbp_features and labels of the testing dataset to a CSV file
output_test_csv_path = os.path.join(output_dir,f"test_lbp_cropped_features_{z-1}.csv")
test_lbp_features_df.to_csv(output_test_csv_path, index=False)
print(f"Testing lbp_features saved to {output_test_csv_path}")

# Save lbp_features and labels of the training dataset to a CSV file
output_train_csv_path = os.path.join(output_dir,f"train_lbp_cropped_labels_{z-1}.csv")
train_labels_df.to_csv(output_train_csv_path, index=False)
print(f"Training lbp_features saved to {output_train_csv_path}")

# Save lbp_features and labels of the training dataset to a CSV file
output_test_csv_path = os.path.join(output_dir,f"test_lbp_cropped_labels_{z-1}.csv")
test_labels_df.to_csv(output_test_csv_path, index=False)
print(f"Training lbp_features saved to {output_train_csv_path}")


# output_dir = "E:\DATA PREPROCESS\csv"
output_csv = "E:\MACHINE LEARNING\CSV_Final_233"

z=no

# Load training embeddings and labels from CSV
trainX = pd.read_csv(os.path.join(output_csv,f"train_lbp_cropped_features_{z}.csv"))
trainy = pd.read_csv(os.path.join(output_csv,f"train_lbp_cropped_labels_{z}.csv"))

# Load testing embeddings and labels from CSV
testX = pd.read_csv(os.path.join(output_csv,f"test_lbp_cropped_features_{z}.csv"))
testy = pd.read_csv(os.path.join(output_csv,f"test_lbp_cropped_labels_{z}.csv"))

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
print(yhat_train_svm.shape)
print(yhat_test_svm.shape)
print(trainy.shape)
print(testy.shape)
accuracy_train_svm = accuracy_score(trainy, yhat_train_svm)
accuracy_test_svm = accuracy_score(testy, yhat_test_svm)
print('SVM Accuracy: train=%.3f, test=%.3f' % (accuracy_train_svm * 100, accuracy_test_svm * 100))

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

# print(timetot)
 
# pd.DataFrame(timetot).to_csv("E:\MACHINE LEARNING\CSV_Final/lbptime30.csv")