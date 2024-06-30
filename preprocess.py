import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import copy

# define data transformations
transform = transforms.Compose([
    transforms.Resize((250, 250)),  # resize images to 250x250 pixels
    transforms.ToTensor(),  # convert images to pytorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize images with mean and std deviation of imagenet dataset
])

# specify the directory containing the dataset
data_dir = 'Processed_Data'

# load the datasets with the specified transformations
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)  # load training dataset with transformations
valid_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)  # load validation dataset with transformations
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)  # load test dataset with transformations

# set batch size for the dataloaders
batch_size = 32  # number of images to be processed in one iteration


# print dataset sizes for confirmation
print(f"Number of training images: {len(train_dataset)}") # 600
print(f"Number of validation images: {len(valid_dataset)}") # 72
print(f"Number of test images: {len(test_dataset)}") # 315