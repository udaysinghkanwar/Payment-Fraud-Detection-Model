
# Credit Card Fraud Detection

A machine learning pipeline designed to detect fraudulent credit card transactions. This project tackles the issue of **highly imbalanced data** (where fraud cases are rare) using statistical analysis, outlier removal, dimensionality reduction, and various classification algorithms.

## 📋 Table of Contents

* [Overview](https://www.google.com/search?q=%23overview)
* [Dataset](https://www.google.com/search?q=%23dataset)
* [Prerequisites](https://www.google.com/search?q=%23prerequisites)
* [Project Workflow](https://www.google.com/search?q=%23project-workflow)
* [1. Exploration & Scaling](https://www.google.com/search?q=%231-exploration--scaling)
* [2. Handling Imbalance](https://www.google.com/search?q=%232-handling-imbalance)
* [3. Outlier Removal](https://www.google.com/search?q=%233-outlier-removal)
* [4. Dimensionality Reduction](https://www.google.com/search?q=%234-dimensionality-reduction)
* [5. Classification](https://www.google.com/search?q=%235-classification)


* [Results](https://www.google.com/search?q=%23results)

## 🔍 Overview

Credit card fraud detection is challenging because fraudulent transactions represent a tiny fraction of total transactions. This script performs the following:

1. Analyzes the original class distribution.
2. Scales time and amount features.
3. Creates a balanced sub-sample (Undersampling) to analyze correlations.
4. Removes extreme outliers from highly correlated features.
5. Visualizes clusters using t-SNE, PCA, and Truncated SVD.
6. Trains and evaluates four different classifiers.

## 💾 Dataset

The script requires a file named `creditcard.csv` in the working directory.

* **Structure:** The dataset should contain features `V1` through `V28` (PCA transformed features), `Time`, `Amount`, and `Class`.
* **Target:** `Class` (0 = No Fraud, 1 = Fraud).

> **Note:** This code is likely built for the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## ⚙️ Prerequisites

To run this project, you need Python installed along with the following libraries:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn

```

## 🚀 Project Workflow

### 1. Exploration & Scaling

* **Data Check:** Calculates the percentage of Fraud vs. Non-Fraud transactions.
* **Scaling:** Uses `RobustScaler` on the `Amount` and `Time` columns. `RobustScaler` is chosen because it is less prone to outliers compared to `StandardScaler`.

### 2. Handling Imbalance (Undersampling)

Because the original data is heavily skewed, we create a **sub-sample** with a 50/50 ratio of Fraud and Non-Fraud cases.

* **Technique:** Random Undersampling.
* **Goal:** To help the model learn the characteristics of fraud cases without being overwhelmed by the majority class.
* **Visualization:** Plots the new balanced distribution.

### 3. Outlier Removal

The script analyzes the correlation matrix of the balanced dataset to identify features heavily correlated with fraud (Positive: V2, V4, V11, V19 | Negative: V10, V12, V14, V17).

* **Method:** Interquartile Range (IQR).
* **Threshold:** Data points beyond `1.5 * IQR` are removed to improve model generalization.
* **Features Cleaned:** V14, V12, and V10.

### 4. Dimensionality Reduction

To visualize the separation between Fraud and Non-Fraud transactions in 2D space, the script employs three techniques:

* **t-SNE:** (t-Distributed Stochastic Neighbor Embedding)
* **PCA:** (Principal Component Analysis)
* **Truncated SVD:** (Singular Value Decomposition)

### 5. Classification

The processed sub-sample is split into training and testing sets (80/20). The following classifiers are trained and evaluated using 5-Fold Cross-Validation:

1. **Logistic Regression**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Classifier (SVC)**
4. **Decision Tree Classifier**

## 📊 Results

Upon execution, the script prints the accuracy scores for the classifiers. Based on the architecture, the expected output format is:

```text
Classifiers:  LogisticRegression Has a training score of XX.XX % accuracy score
Classifiers:  KNeighborsClassifier Has a training score of XX.XX % accuracy score
Classifiers:  SVC Has a training score of XX.XX % accuracy score
Classifiers:  DecisionTreeClassifier Has a training score of XX.XX % accuracy score

```

*Visualizations generated during runtime:*

1. **Class Distribution Bar Chart** (Original vs Balanced).
2. **Correlation Heatmap.**
3. **Box Plots** for negative and positive correlations.
4. **Cluster Scatter Plots** (t-SNE, PCA, SVD).

---
