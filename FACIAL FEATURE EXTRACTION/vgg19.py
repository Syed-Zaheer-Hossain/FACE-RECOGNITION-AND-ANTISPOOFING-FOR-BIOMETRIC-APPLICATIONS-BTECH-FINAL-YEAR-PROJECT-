from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
import numpy as np
import keras
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import time

# Load pre-trained VGG19 model
vgg_model = VGG19(weights='imagenet', include_top=False, pooling='avg')

# Define a function to extract features using the VGG19 model
def extract_vgg_features(image_path):
    img = Image.open(image_path)
    img = img.resize((100, 100))  
    x = np.array(img)
    x = preprocess_input(x)  
    x = np.expand_dims(x, axis=0)
    features = vgg_model.predict(x)
    return features.flatten()

# Path to the main directory containing numerical-named folders
# base_dir = r'E:\DataSet'
base_dir = r'E:\MACHINE LEARNING\output2'

csv = "E:\MACHINE LEARNING\CSV_Final_100"

# Initialize lists to store face embeddings and corresponding labels for training and testing
train_embeddings = []
train_labels = []
test_embeddings = []
test_labels = []

z=1
no=152
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
output_train_csv_path = os.path.join(csv,f'train_embeddings_and_labels_{crop}_{no}.csv')
output_test_csv_path = os.path.join(csv,f'test_embeddings_and_labels_{crop}_{no}.csv')

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
train_embeddings_df = pd.read_csv('E:\PROJECT UPLOADS GITHUB/train_embeddings_and_labels.csv')  # Adjust path
trainX = train_embeddings_df.drop(columns=['label']).values
trainy = train_embeddings_df['label'].values

# Load testing embeddings and labels from CSV
test_embeddings_df = pd.read_csv('E:\PROJECT UPLOADS GITHUB/test_embeddings_and_labels.csv')  # Adjust path
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