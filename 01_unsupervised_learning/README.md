# Unsupervised Learning Project

## Overview
This project explores three fundamental unsupervised learning techniques applied to two distinct datasets. The focus is on understanding patterns, reducing dimensionality, and identifying clusters in data without labeled examples.

## Datasets

### 1. Temperature Dataset (temper.npz)
- Monthly temperature data for 15 French cities
- Features: 15×12 (cities × months)
- Additional data: latitude/longitude coordinates

### 2. Digits Dataset (digits.npz)
- Handwritten digit images (1, 7, 8)
- Dimensions: 3000×784 (samples × pixels)
- 28×28 pixel images normalized to [0,1]

## Methods Implemented

### 1. Clustering
- K-Means clustering with optimal K selection
  - Elbow method analysis
  - Silhouette score evaluation
- Gaussian Mixture Models (GMM)
  - Probabilistic clustering
  - Outlier detection

### 2. Density Estimation
- GMM for probability density estimation
- Generation of synthetic data
- Outlier detection capabilities

### 3. Dimensionality Reduction
- Principal Component Analysis (PCA)
  - Linear projection
  - Data compression
  - Variance explained analysis
- t-SNE
  - Non-linear manifold learning
  - Visualization of high-dimensional data
  - Perplexity parameter analysis

## Key Findings

### Temperature Data
- Optimal clustering reveals 3 distinct climate zones in France
- PCA shows >90% variance explained with 2 components
- t-SNE effectively visualizes city clusters based on climate similarity

### Digits Data
- Successful separation of digit classes using both clustering methods
- PCA requires ~73 components for 90% variance
- t-SNE provides superior visualization compared to PCA

## Technologies Used
- Python
- NumPy
- Scikit-learn
- Matplotlib
- SciPy

## Project Structure
```
/03_classification/
├── data/
│   ├── temper.npz
│   └── digits.npz
├── notebooks/
│   └── Lab1.ipynb
└── README.md
```

## Usage
1. Load the datasets using numpy:
```python
temper_data = np.load("data/temper.npz")
digits_data = np.load("data/digits.npz")
```

2. Follow the notebook for detailed analysis and implementation

## Requirements
- Python 3.x
- Required packages listed in requirements.txt
