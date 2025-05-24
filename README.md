
# 🧾 Banknote Authentication using Logistic Regression

**Author:** Victoria Tanoh  
**Specialization:** Data Science | Finance & Audit Automation | Graph & Statistical Modeling

---

## 📘 Project Overview

This project applies machine learning to distinguish between **authentic and forged banknotes** using statistical image features. It showcases a full workflow — from **data acquisition** and **exploratory data analysis (EDA)** to **model building** and **visualization** — serving as a practical example for financial fraud detection and document authentication.

The dataset originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication), offering an ideal entry point for anomaly detection in financial documents.

---

## 📊 Dataset Details

- **Source:** UCI ML Repository – Banknote Authentication Dataset (ID: 267)
- **Features:**
  - `variance`: Variance of wavelet-transformed image
  - `skewness`: Skewness of wavelet-transformed image
  - `curtosis`: Kurtosis of wavelet-transformed image
  - `entropy`: Entropy of image
- **Target:**
  - `0` = Authentic banknote
  - `1` = Forged banknote

---

##  Project Highlights

- 🧾 **Data fetching** using the `ucimlrepo` package
- 📊 **EDA** using pairplots and correlation heatmaps
- 🧠 **Logistic Regression** model for binary classification
- ✅ **Evaluation** using accuracy, confusion matrix, F1-score
- 🔍 **PCA visualization** for data separability

---

## 💻 Installation & Usage

### 🔧 Requirements

```bash
pip install ucimlrepo scikit-learn pandas matplotlib seaborn

