# LungAI

LungAI is a deep learning-based lung cancer detection application that runs in the browser. This project uses a custom Convolutional Neural Network model built with PyTorch to classify lung cancer images.

4x hackathon award winner - out of 1,500 total competitors.

The lung cancer detection model achieves 98% accuracy in distinguishing between cancerous and non-cancerous cases, while maintaining 85% accuracy in differentiating between four specific types of lung conditions: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, and normal (non-cancerous) tissue.

<i> This project represents the newest version, now using PyTorch.</i>


[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/DorsaRoh/LungAI)
[![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFFF00?style=for-the-badge&logo=huggingface)](https://huggingface.co/dorsar/LungAI)

## Project Structure

- `Model/`
  - `architecture.py`: Contains the custom CNN model architecture and the training script.
  - `metadata.py`: Handles dataset processing and metadata extraction.
  - `convert_to_onnx.py`: Converts the trained PyTorch model to ONNX format for browser compatibility.
- `index.html`: The front-end interface for running the application in the browser.
- `requirements.txt`: Lists the Python dependencies required for this project.

## Setup and Usage

### Step 1: Install Dependencies

First, ensure you have Python installed. Then, install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```

### Step 1: Install Dependencies

First, ensure you have Python installed. Then, install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

Run the training script to train the custom CNN model on your lung cancer dataset:

```bash
python Model/architecture.py
```

### Step 3: Process Metadata

If you need to preprocess your dataset and extract metadata, run the metadata script:

```bash
python Model/metadata.py
```

### Step 4: Convert the Model to ONNX

Convert the trained PyTorch model to ONNX format for compatibility with web applications:

```bash
python Model/convert_to_onnx.py
```

### Step 5: Run the Application in the Browser

Open `index.html` in your preferred web browser to start using the LungAI application.

```bash
<!-- Open this file in your browser -->
index.html
```

### Notes

- Make sure your dataset is structured correctly under the Processed_Data directory with subdirectories for training, validation, and testing sets.
- The model training script expects the dataset to be in the Processed_Data directory. Ensure that the data transformations and directory paths are correctly set up in architecture.py.
- The ONNX model conversion script (convert_to_onnx.py) requires the trained model file. Ensure you have successfully trained the model before running this script.

### Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome improvements, bug fixes, and new features.

## Connect with Me

[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/DorsaRoh)
[![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?style=for-the-badge&logo=twitter)](https://twitter.com/Dorsa_Rohani)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/dorsarohani/)