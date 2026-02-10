# -*- coding: utf-8 -*- 
"""
Created on Mon Sep  4 20:53:03 2023

"""

import os
import cv2
from mtcnn import MTCNN
import pandas as pd
import time

# Create the MTCNN detector
mtcnn_detector = MTCNN(steps_threshold=[0.7, 0.8, 0.8])  

# Define input and output directories
# master_dir = input("Enter the master folder address\n\n")
master_dir = "E:\single"
original = "fake"
preprocess = "preprocessed-fake"

#code for calculation of time

start = time.time()
timetot=[]
timet = ["No. of images" , "Time Taken"]
timetot.append(timet)

counter = add = 10
j=0

for foldername in os.listdir(master_dir):
    
    if(not os.path.exists(os.path.join(master_dir,preprocess))):
        os.makedirs(os.path.join(master_dir,preprocess))
        
    # Renaming folders with original and fake folders named in uppercase
    
    if(os.path.exists(os.path.join(master_dir,foldername,"Original"))):
       os.rename(os.path.join(master_dir,foldername,"Original"), os.path.join(master_dir,foldername,"original"))

    if(os.path.exists(os.path.join(master_dir,foldername,"Fake"))):
       os.rename(os.path.join(master_dir,foldername,"Fake"), os.path.join(master_dir,foldername,"fake"))    
    
    if(foldername != "preprocessed"):
        
        for filename in os.listdir(os.path.join(master_dir,foldername,original)):
            
            # CODE FOR CROPPING THE FACE FROM IMAGES USING MTCNN PYTHON PACKAGE
            
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                input_image_path = os.path.join(master_dir,foldername,original, filename)
                
                # Load the input image
                input_image = cv2.imread(input_image_path)
                # print(input_image.shape)
                input_image = cv2.resize(input_image,(500,600))
        
                # Convert the image to RGB (MTCNN expects RGB images)
                input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
                # Detect faces in the input image
                faces = mtcnn_detector.detect_faces(input_image_rgb)
        
                # Process each detected face
                for i, face_info in enumerate(faces):
                    box = face_info['box']
                    x, y, w, h = box
        
                    # Crop the face from the input image
                    extracted_face = input_image[y:y+h, x:x+w]
        
                    if(i==0):
                        output_face_path = os.path.join(master_dir, foldername,original, f"{filename[:(len(filename)-4)]}.png")
                    else:
                        output_face_path = os.path.join(master_dir, foldername,original, f"{filename[:(len(filename)-4)]}_face{i+1}.png")
        
                    cv2.imwrite(output_face_path, extracted_face)
                # code for execution of time calculation
                
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
                
                    
        print("Face extraction complete for folder - " + foldername)
        
        
                    
        # CODE FOR STORING THE PNG CROPPED FILES INTO ANOTHER FOLDER
        
        for filename in os.listdir(os.path.join(master_dir,foldername,original)):
            
            if filename.lower().endswith(".png"):
                img_path = os.path.join(master_dir,foldername,original,filename)
                img = cv2.imread(img_path)           #method 1
                img = cv2.resize(img,[250,300])      #method 1
                output_dir = os.path.join(master_dir,preprocess,foldername)
                if(not os.path.exists(output_dir)):
                    os.makedirs(output_dir)
                output_addr = os.path.join(output_dir,filename)
                # shutil.copy(img_path, output_addr)   #method 2
                cv2.imwrite(output_addr, img)        #method 1
        print("Seperate Preproccessing data created for folder-"+foldername)

end = time.time()
tott = end - start

timet=[]
timet.append(j)
timet.append(tott)
timetot.append(timet)
 
pd.DataFrame(timetot).to_csv("E:\MACHINE LEARNING\IMAGE PROCESSING/mtcnntime3.csv")

print("the total time taken is " + str(tott))
print("the total time array is " + str(timetot))