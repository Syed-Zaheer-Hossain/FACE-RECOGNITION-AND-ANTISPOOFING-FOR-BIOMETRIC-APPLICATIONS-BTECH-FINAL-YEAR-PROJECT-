# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:08:57 2024

@author: SYED ZAHEER HOSSAIN
"""

import os
import cv2
from mtcnn import MTCNN

# Create the MTCNN detector
mtcnn_detector = MTCNN(steps_threshold=[0.7, 0.8, 0.8])

# Define input and output directories
input_master_dir = r"E:/dataset all combined/comb"
output_master_dir = r"E:\MACHINE LEARNING\output233"

os.makedirs(output_master_dir)

# Loop through folders named from 1 to 10
for folder_index in range(1, 233):
    folder_name = f"{folder_index:04}"
    
    # Loop through fake and original folders
    for sub_folder in ["fake", "original"]:
        input_folder_path = os.path.join(input_master_dir, folder_name, sub_folder)
        output_folder_path = os.path.join(output_master_dir, folder_name, sub_folder)
        
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

        # Process images in the current folder
        for filename in os.listdir(input_folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                input_image_path = os.path.join(input_folder_path, filename)
                output_image_path = os.path.join(output_folder_path, filename)

                # Load the input image
                input_image = cv2.imread(input_image_path)
                input_image = cv2.resize(input_image,(400,500))
                input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

                # Detect faces in the input image
                faces = mtcnn_detector.detect_faces(input_image_rgb)

                # Process each detected face
                for i, face_info in enumerate(faces):
                    box = face_info['box']
                    x, y, w, h = box

                    # Crop the face from the input image
                    extracted_face = input_image[y:y+h, x:x+w]

                    # Save the cropped face
                    if i == 0:
                        output_face_path = output_image_path[:-4] + ".png"
                    else:
                        output_face_path = output_image_path[:-4] + f"{i + 1}.png"

                    cv2.imwrite(output_face_path, extracted_face)

        print("Face extraction complete for folder:", folder_name, sub_folder)

print("All folders processed successfully.")