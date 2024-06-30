import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# data transformations with augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ResNetLungCancer(nn.Module):
    def __init__(self, num_classes, use_pretrained=True):
        super(ResNetLungCancer, self).__init__()
        if use_pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.resnet = resnet50(weights=weights)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # remove the final fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return self.fc(x)


# train function
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
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

            if phase == 'valid':
                scheduler.step(epoch_acc)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Learning rate: {current_lr}')
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# eval the model
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Acc: {test_acc:.4f}')

if __name__ == "__main__":
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # data
    data_dir = 'Processed_Data'
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    valid_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_test_transform)

    #  dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(valid_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")

    # initialize model, loss, and optimizer
    num_classes = len(train_dataset.classes)
    model = ResNetLungCancer(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    pretrained_params = list(model.resnet.parameters())
    new_params = list(model.fc.parameters())

    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': 1e-5},
        {'params': new_params, 'lr': 1e-4}
    ], weight_decay=1e-6)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    # train the model
    trained_model = train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=50, device=device)

    # eval the model
    evaluate_model(trained_model, test_loader, device=device)

    # save the model weights
    torch.save(trained_model.state_dict(), 'lung_cancer_detection_model.pth')

    # save the model in ONNX format
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(trained_model, dummy_input, "lung_cancer_detection_model.onnx", input_names=['input'], output_names=['output'])

    print("Training completed. Model saved.")