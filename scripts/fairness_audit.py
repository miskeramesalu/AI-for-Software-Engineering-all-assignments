# =========================================================
# fairness_audit.py
# =========================================================
# Purpose: Basic fairness audit template for a classification model
# Author: Misker Amesalu Mulu
# =========================================================

# ---------------------------
# 1. Imports
# ---------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Fairness package (install via: pip install fairlearn)
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate

# ---------------------------
# 2. Load dataset
# ---------------------------
# Replace 'data.csv' with your dataset file
data = pd.read_csv("data.csv")

# Example: Assume 'target' is label and 'gender' is sensitive feature
target_column = 'target'
sensitive_column = 'gender'

X = data.drop(columns=[target_column])
y = data[target_column]
sensitive_features = data[sensitive_column]

# ---------------------------
# 3. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
    X, y, sensitive_features, test_size=0.3, random_state=42
)

# ---------------------------
# 4. Preprocessing
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 5. Train a simple model
# ---------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# ---------------------------
# 6. Model evaluation
# ---------------------------
print("=== Model Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# 7. Fairness audit
# ---------------------------
metric_frame = MetricFrame(
    metrics={
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sf_test
)

print("\n=== Fairness Metrics by Group ===")
print(metric_frame.by_group)

print("\n=== Overall Fairness Metrics ===")
print(metric_frame.overall)