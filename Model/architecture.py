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

# create dataloaders for the datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # create dataloader for training data with shuffling
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)  # create dataloader for validation data without shuffling
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # create dataloader for test data without shuffling

# print dataset sizes for confirmation
#print(f"Number of training images: {len(train_dataset)}") # 600
#print(f"Number of validation images: {len(valid_dataset)}") # 72
#print(f"Number of test images: {len(test_dataset)}") # 315

# define the CNN architecture
class LungCancerCNN(nn.Module):
    def __init__(self, num_classes):
        super(LungCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 31 * 31, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 31 * 31)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = len(train_dataset.classes)
model = LungCancerCNN(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=25)

def evaluate_model(model, test_loader):
    model.eval()
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Acc: {test_acc:.4f}')

evaluate_model(trained_model, test_loader)
torch.save(trained_model.state_dict(), 'lung_cancer_detection_model.pth')
