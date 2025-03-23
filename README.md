# Breast-Cancer-Classification-using-Custom-Neural-Network

![Breast Cancer Classification](https://github.com/AshishRaj04/Breast-Cancer-Classification-using-Custom-Neural-Network/blob/main/results/attribute_histogram_plots.png)

## Project Overview
This project focuses on classifying breast cancer tumors as Malignant (cancerous) or Benign (non-cancerous) using a custom neural network. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset from Kaggle.

While the dataset suggests using **SVM**, this project aims to test the performance of a custom-built deep learning model. The model is trained and evaluated using Google Colab to leverage cloud-based computation.

## Folder Structure


/breast_cancer_classification
│── data/ # Store dataset
│── notebooks/ # Store Colab notebooks
│── src/ # Python scripts (model, preprocessing)
│── models/ # Saved models
│── results/ # Evaluation metrics, plots
│── requirements.txt # Dependencies
│── README.md # Project overview
│── .gitignore # Ignore unnecessary files
│── images/ # Store project-related images 

## Dataset

- **Name**: Breast Cancer Wisconsin (Diagnostic) Dataset

- **Source**: [Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data)


## Model Architecture

- **Input Layer**: 30 features
- **Hidden Layers**: Fully connected layers with ReLU activation
- **Output Layer**: Binary classification (Malignant/Benign) with Sigmoid activation
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

## Setup & Installation

### 1. Clone Repository in Colab
``` bash
!git clone <repo_url>
%cd breast_cancer_classification
```
### 2. Install Dependencies
``` bash 
!pip install -r requirements.txt
```
## Results & Performance

- Model Accuracy: *To be updated after training*
- Precision, Recall, F1-score: *To be updated*
- Comparison with SVM: *To be analyzed*

## Contributors
- Ashish Raj Prasad

## License
MIT License
