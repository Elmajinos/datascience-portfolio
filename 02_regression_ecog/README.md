# ECoG Regression Analysis Project

## Overview
This project focuses on applying machine learning regression techniques to predict thumb flexion from ECoG (Electrocorticography) signals. The dataset comes from BCI Competition IV, specifically analyzing subject 3's brain signals.

## Dataset
- ECoG signals from multiple electrodes
- Target variable: thumb flexion measurements
- Training set: 1000 samples
- Test set: Remaining samples
- Sampling frequency (Fe) provided

## Methods Implemented

### 1. Linear Models
- Least Squares Regression (LS)
  - Basic implementation
  - Feature selection variant
- Ridge Regression
  - Regularization parameter optimization
  - Coefficient analysis
- Lasso Regression
  - Feature selection capabilities
  - Optimal alpha determination

### 2. Non-linear Models
- Support Vector Regression (SVR)
  - Kernel selection
  - Hyperparameter tuning
- Random Forest
  - n_estimators optimization
  - max_depth analysis
  - min_samples_split tuning
- Multi-Layer Perceptron (MLP)
  - Architecture optimization
  - Learning rate tuning
  - Regularization parameter selection

## Key Findings

### Linear Models
- Least Squares: Severe overfitting (R²_train = 0.84, R²_test = -0.43)
- Ridge: Better generalization with α = 954
- Lasso: Best performing linear model (α = 0.132)
  - Selected 14 important features
  - R²_test = 0.36

### Non-linear Models
- Random Forest: Good performance with optimized parameters
- SVR: Comparable to linear models
- MLP: Improved performance after optimization
  - Two hidden layers (50, 50)
  - Learning rate = 0.0001

## Best Model Performance
Lasso Regression (α = 0.132):
- MSE Test: 1.368
- R² Test: 0.358
- Features used: 14 selected electrodes

## Project Structure
```
/02_regression_ecog/
├── data/
│   └── ECoG.npz
├── notebooks/
│   └── Lab2.ipynb
└── README.md
```

## Technologies Used
- Python
- NumPy
- Scikit-learn
- Matplotlib
- Pandas

## Usage
1. Load the ECoG dataset:
```python
data = np.load('data/ECoG.npz')
X = data['Xall']
Yall = data['Yall']
```

2. Follow the notebook for detailed analysis and implementation

## Requirements
- Python 3.x
- Required packages listed in requirements.txt

## Medical Applications
This project demonstrates the potential of machine learning in neural signal processing, with particular emphasis on:
- Signal interpretation
- Feature selection
- Model interpretability
- Clinical applicability

## Conclusions
- Lasso regression provides the best balance of performance and interpretability
- Linear models perform surprisingly well for this application
- Feature selection is crucial for model performance
- Careful validation is essential for medical applications
