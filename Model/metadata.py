import json
from architecture import num_classes

metadata = {
    "architecture": "LungCancerCNN",
    "num_classes": num_classes,
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f)
