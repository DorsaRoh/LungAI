import torch
from PIL import Image
from torchvision import transforms
from architecture import ResNetLungCancer
import gradio as gr

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

class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

def predict(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    predicted_class = torch.argmax(output, dim=1).item()
    return class_names[predicted_class]

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=1),
    examples=[
        ["path/to/example/image1.png"],
        ["path/to/example/image2.png"]
    ]
)

iface.launch()