import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# define the model architecture
class ResNetLungCancer(nn.Module):
    def __init__(self, num_classes=4, use_pretrained=True):
        super(ResNetLungCancer, self).__init__()
        if use_pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.resnet(x)
        return self.fc(x)

# define data transformations
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# test the model
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# main execution
if __name__ == "__main__":
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the model
    model = ResNetLungCancer(num_classes=4)
    model.load_state_dict(torch.load('lung_cancer_detection_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # prepare the test dataset
    test_dir = 'Processed_Data/test' 
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Number of test images: {len(test_dataset)}")
    print(f"Number of classes: {len(test_dataset.classes)}")
    print(f"Classes: {test_dataset.classes}")

    # test the model
    test_model(model, test_loader, device)

    print("Testing completed.")