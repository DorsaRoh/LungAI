


# preprocess data

    # extra:
        # remove noise
        # filters: gaussian filter? median filter? bilateral filter? non local means denoising? anisotropic diffusion?
        # increase contrast
        # remove non-lung parts of iamge


    # resize to 250 x 250 pixel



import pydicom
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path to data and csv file
data_dir = 'data/manifest-1719591652216/NSCLC-Radiomics'
csv_file = 'data/manifest-1719591652216/metadata.csv'

# read the csv file
metadata = pd.read_csv(csv_file)

# load a dicom file
def load_dicom(dicom_path):
    try:
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array
        return image
    except Exception as e:
        print(f"Error reading DICOM file {dicom_path}: {e}")
        return None

# display a dicom image
def display_dicom(image):
    if image is not None:
        plt.imshow(image, cmap='gray')
        plt.show()
    else:
        print("No image to display")

# iterate over the rows of the csv file and process the dicom files
for index, row in metadata.iterrows():
    subject_id = row['Subject ID'] 
    lung_dir = os.path.join(data_dir, subject_id)
    
    if os.path.isdir(lung_dir):
        for root, _, files in os.walk(lung_dir):
            for filename in files:
                if filename.endswith(".dcm"):
                    dicom_path = os.path.join(root, filename)
                    print(f"Reading DICOM file: {dicom_path}")
                    dicom_image = load_dicom(dicom_path)
                    
                    if dicom_image is not None:
                        print(f"Image shape: {dicom_image.shape}")
                    
                    # display the dicom image
                    display_dicom(dicom_image)
                    
                    

    else:
        print(f"Directory not found: {lung_dir}")

# Note: add the code to prepare your data for the deep learning model, such as converting to arrays, batching, etc.
