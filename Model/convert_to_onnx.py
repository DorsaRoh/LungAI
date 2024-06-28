import torch
import onnx
from model import LungCancerCNN  # ensure this imports your model definition

# load your trained model
model_path = 'lung_cancer_detection_model.pth'
num_classes = 4  
model = LungCancerCNN(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

input = torch.randn(1, 3, 250, 250)

# export the model to ONNX format
onnx_model_path = 'lung_cancer_detection_model.onnx'
torch.onnx.export(model, input, onnx_model_path, 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
print(f'Model successfully converted to {onnx_model_path}')
