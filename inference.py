import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os

class ResNetLungCancer(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetLungCancer, self).__init__()
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

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetLungCancer(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def load_class_names():
    class_names = {
        0: "Adenocarcinoma",
        1: "Large Cell Carcinoma",
        2: "Normal",
        3: "Squamous Cell Carcinoma"
    }
    return class_names

def main(image_path):
    model_path = os.path.join('Model', 'lung_cancer_detection_model.pth')

    model, device = load_model(model_path)
    image_tensor = preprocess_image(image_path)
    prediction = predict(model, image_tensor, device)
    
    class_names = load_class_names()
    predicted_class = class_names[prediction]

    return {
        "prediction": prediction,
        "predicted_class": predicted_class
    }

if __name__ == "__main__":
    # For testing purposes
    test_image_path = "path/to/test/image.jpg"
    result = main(test_image_path)
    print(f"Prediction: {result['prediction']}")
    print(f"Predicted Class: {result['predicted_class']}")