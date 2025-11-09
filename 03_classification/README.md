# Supervised Classification Project

## Overview
This project implements and compares various supervised classification methods on two distinct datasets: the Pima Indians Diabetes dataset and the Handwritten Digits dataset. The focus is on understanding model performance, interpretability, and robustness.

## Datasets

### 1. Pima Indians Diabetes Dataset
- Medical data for diabetes prediction
- 8 features (Glucose, BMI, Blood Pressure, etc.)
- Binary classification task (diabetic/non-diabetic)
- Training set: 300 samples
- Test set: remaining samples

### 2. Digits Dataset
- Handwritten digits (1, 7, 8)
- 28×28 pixel images
- Multi-class classification
- Normalized pixel values [0,1]

## Methods Implemented

### 1. Linear Classifiers
- Linear Discriminant Analysis (LDA)
  - Shrinkage parameter optimization
  - Intercept adjustment for medical context
- Logistic Regression
  - L1/L2 regularization
  - Feature importance analysis

### 2. Non-linear Classifiers
- Support Vector Machines (SVM)
  - Kernel selection
  - Hyperparameter tuning
- Random Forest
  - Parameter optimization
- Neural Networks
  - MLP Classifier
  - Convolutional Neural Network (PyTorch)

## Key Findings

### Pima Dataset
- Best model: LDA with shrinkage=0.5
  - ROC AUC: 0.869
  - Accuracy: 0.814
  - F1-Score: 0.686
- Most important features: Glucose, BMI, DiabetesPedigreeFunction
- Linear models sufficient for this medical application

### Digits Dataset
- Best model: SVM with RBF kernel
  - Accuracy: 0.991
  - Excellent class separation
- CNN shows superior robustness to noise
- High performance across all models due to well-separated classes

## Project Structure
```
/03_classification/
├── data/
│   ├── pima.npz
│   └── digits.npz
├── notebooks/
│   └── Lab3.ipynb
└── README.md
```

## Technologies Used
- Python
- NumPy
- Scikit-learn
- PyTorch
- Pandas
- Matplotlib

## Usage
1. Load the datasets:
```python
pima = np.load("data/pima.npz")
digits = np.load("data/digits.npz")
```

2. Follow the notebook for detailed analysis

## Medical Applications
- Model interpretation focus
- False Negative Rate optimization
- Risk assessment trade-offs
- Robust prediction requirements

## Requirements
- Python 3.x
- Required packages in requirements.txt
- CUDA support (optional, for CNN)
