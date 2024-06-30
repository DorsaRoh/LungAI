import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models import resnet50, ResNet50_Weights

# Define the model architecture (same as in your training script)
class ResNetLungCancer(nn.Module):
    def __init__(self, num_classes=4, use_pretrained=True):
        super(ResNetLungCancer, self).__init__()
        if use_pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.resnet = resnet50(weights=weights)
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

# Define data transformations (same as in your training script)
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    data_dir = 'Processed_Data'  # Update this to your test data directory
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

    # Create dataloader
    batch_size = 32
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Number of test images: {len(test_dataset)}")

    # Initialize model
    num_classes = len(test_dataset.classes)
    model = ResNetLungCancer(num_classes)

    # Load trained model weights
    model.load_state_dict(torch.load('lung_cancer_detection_model.pth'))
    model = model.to(device)

    # Evaluate the model
    evaluate_model(model, test_loader, device)

    print("Testing completed.")