# Breast-Cancer-Classification-using-Custom-Neural-Network

![Breast Cancer Classification](https://github.com/AshishRaj04/Breast-Cancer-Classification-using-Custom-Neural-Network/blob/main/results/loss vs epoch fro df_transformed.png)

## Project Overview
This project focuses on classifying breast cancer tumors as Malignant (cancerous) or Benign (non-cancerous) using a custom neural network. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset from Kaggle.

While the dataset suggests using **SVM**, this project aims to test the performance of a custom-built deep learning model. The model is trained and evaluated using Google Colab to leverage cloud-based computation.

## Dataset

- **Name**: Breast Cancer Wisconsin (Diagnostic) Dataset

- **Source**: [Kaggle](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data)


## Model Architecture

- **Input Layer**: 10 features
- **Hidden Layers**: Fully connected layers with tanh activation
- **Output Layer**: Binary classification (Malignant/Benign) with Sigmoid activation
- **Loss Function**: MSE
- **Optimizer**: stochastic gradient descent
- **Evaluation Metrics**: Accuracy

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

- Model Accuracy: 70.18% on test data
- Comparison with SVM: *To be analyzed*

## Conclusion

This project investigated the development of a custom neural network for breast cancer classification. Two data preprocessing strategies were evaluated: utilizing df_combined, which involved MinMax scaling of numerical features, and utilizing df_transformed, which incorporated a logarithmic transformation of select features (perimeter_mean, area_mean, compactness_mean, concavity_mean, concave points_mean) following MinMax scaling.

Remarkably, both approaches achieved a test accuracy of 70.18%, suggesting that the logarithmic transformation did not significantly influence the model's performance for this specific dataset. This outcome was somewhat unexpected, as the transformation aimed to address skewness in certain features.

While the achieved accuracy is encouraging, the training process exhibited relatively high loss values, even after 30 epochs. This observation suggests potential for improvement in the model's training dynamics. Further investigation could involve exploring alternative loss functions or optimization algorithms to potentially enhance training efficiency and reduce loss.

Despite the observed training behavior, the model demonstrated robust performance on the test set, achieving a commendable accuracy of 70.18%. This outcome highlights the effectiveness of the custom neural network architecture in capturing the underlying patterns in the data and generalizing well to unseen instances.

Future work can build upon this foundation by focusing on:

- **Hyperparameter tuning**: Investigating different network architectures, activation functions, and learning rates to potentially boost performance further.

- **Feature engineering**: Exploring new features or transformations that could better capture the underlying patterns in the data.

In conclusion, this project demonstrated the feasibility of constructing a custom neural network for breast cancer classification with high accuracy. Further research can refine this approach to develop even more robust and accurate models for this critical application.


## Contributors
- Ashish Raj Prasad

## License
MIT License
