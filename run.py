import torch
from PIL import Image
from torchvision import transforms
from architecture import ResNetLungCancer 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNetLungCancer(num_classes=4)
model.load_state_dict(torch.load('Model/lung_cancer_detection_model.pth', map_location=device))
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# image from local file
image_path = "Data/test/large.cell.carcinoma/000108.png"
image = Image.open(image_path).convert('RGB') 

# preprocess the image
input_tensor = preprocess(image).unsqueeze(0).to(device)  # add batch dimension and move to device

# get model predictions
with torch.no_grad():
    output = model(input_tensor)

predicted_class = torch.argmax(output, dim=1).item()

class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

print(f"Predicted class: {class_names[predicted_class]}")