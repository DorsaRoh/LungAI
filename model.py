import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image, apply_full_preprocessing=True):
    # convert image to grayscale if not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = enhance_contrast(image)
    image = resize_image(image)
    image = standardize_image(image)
    
    return image

# contrast enhancement
def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    return image

# resize image
def resize_image(image, size=(250, 250)):
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image

# standardize image to mean 0 and std 1, then scale to 0-255 range
def standardize_image(image):
    image = (image - np.mean(image)) / np.std(image)
    image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255
    return image

# process all images in a directory
def process_directory(directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(subdir, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                preprocessed_image = preprocess_image(image)
                
                # save preprocessed image
                relative_path = os.path.relpath(subdir, directory)
                output_path = os.path.join(output_directory, relative_path)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                
                output_image_path = os.path.join(output_path, file)
                cv2.imwrite(output_image_path, preprocessed_image.astype(np.uint8))
                
                # display the preprocessed image
                # plt.imshow(preprocessed_image, cmap='gray')
                # plt.title(f'Preprocessed Image: {file}')
                # plt.show()

# paths to train, valid, and test directories
train_dir = 'Data/train'
valid_dir = 'Data/valid'
test_dir = 'Data/test'

# output directories for preprocessed images
train_output_dir = 'Processed_Data/train'
valid_output_dir = 'Processed_Data/valid'
test_output_dir = 'Processed_Data/test'

# apply preprocessing
process_directory(train_dir, train_output_dir)
process_directory(valid_dir, valid_output_dir)
process_directory(test_dir, test_output_dir)
