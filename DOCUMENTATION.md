
# 📄 Documentation: Banknote Authentication using Logistic Regression

## 📘 Overview

This project uses logistic regression to classify whether a banknote is **authentic or forged** based on image statistics from the UCI Machine Learning Repository.

---

## 🧠 Python Code Breakdown

### 1. Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
```

These libraries are used for data manipulation, visualization, modeling, evaluation, and feature scaling.

---

### 2. Load the Dataset

```python
data = fetch_ucirepo(id=267)
X = data.data.features
y = data.data.targets
```

This fetches the **Banknote Authentication Dataset** using `ucimlrepo`.

---

### 3. Combine DataFrame for EDA

```python
df = X.copy()
df['target'] = y
```

Adds the target column for exploratory analysis.

---

### 4. Exploratory Data Analysis (EDA)

```python
sns.pairplot(df, hue='target', diag_kind='kde')
sns.heatmap(df.corr(), annot=True)
```

- Shows pairwise feature distributions by class
- Shows feature correlations

---

### 5. Train-Test Split and Scaling

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Splits data and standardizes it for consistent model behavior.

---

### 6. Model Training

```python
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

Trains a logistic regression model and predicts test labels.

---

### 7. Evaluation Metrics

```python
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Displays how well the model performed:
- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, F1-score**

---

### 8. PCA Visualization

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler.fit_transform(X))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)
```

Reduces feature space to 2D and visualizes class separability.

---

## ✅ Summary

This script demonstrates:
- Clean dataset handling from UCI
- Proper EDA techniques
- Logistic regression model building
- Visual and metric-based evaluation

It is a great starting point for building document fraud detection systems in financial auditing.

