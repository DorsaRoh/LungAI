# LUNGAI: Lung Cancer Detection Model

## Project Overview
LungAI is a deep learning project aimed at detecting and classifying lung cancer from CT scan images. The model can differentiate between cancerous and non-cancerous lung tissue, as well as classify specific types of lung cancer.

4x hackathon award winner - out of 1,500 total competitors.

[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/DorsaRoh/LungAI)
[![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFFF00?style=for-the-badge&logo=huggingface)](https://huggingface.co/dorsar/lung-cancer-detection)


## Model Performance
- 98% accuracy in distinguishing between cancerous and non-cancerous cases
- 83% accuracy in differentiating between four specific types of lung conditions:
  - Adenocarcinoma: 82% F1-score
  - Large Cell Carcinoma: 85% F1-score
  - Normal (non-cancerous): 98% F1-score
  - Squamous Cell Carcinoma: 76% F1-score

<i>This project represents the newest version, now using PyTorch.</i>

## Repository Structure
- `Architecture/`: Contains the core model scripts
  - `architecture.py`: Defines the model architecture
  - `preprocess.py`: Data preprocessing utilities
  - `test.py`: Script for testing the model
- `Model/`: Stores trained model files
  - `lung_cancer_detection_model.onnx`: ONNX format of the trained model
  - `lung_cancer_detection_model.pth`: PyTorch weights of the trained model
- `Data/`: (Not included in repository) Directory for storing the dataset
- `Processed_Data/`: (Not included in repository) Directory for preprocessed data
- `assets/`: Additional project assets
- `requirements.txt`: List of Python dependencies


## Setup and Usage

### Step 1: Install Dependencies

First, ensure you have Python installed. Then, install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (Optional)

Run the training script to train the model. 
**It will be saved as `.pth` and `.onnx` files**

```bash
python Architecture/architecture.py
```

### Step 3: Run the Model

Run the model by running the following file:

```bash
python Architecture/run.py
```

### Notes

- Make sure your dataset is structured correctly under the Processed_Data directory with subdirectories for training, validation, and testing sets.
- The model training script expects the dataset to be in the Processed_Data directory. Ensure that the data transformations and directory paths are correctly set up in architecture.py.

### Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome improvements, bug fixes, and new features.

## Connect with Me

[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/DorsaRoh)
[![Twitter](https://img.shields.io/badge/-Twitter-1DA1F2?style=for-the-badge&logo=twitter)](https://twitter.com/Dorsa_Rohani)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/dorsarohani/)