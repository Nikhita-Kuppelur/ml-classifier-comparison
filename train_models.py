"""
Train 5 classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM)
on the UCI Breast Cancer Wisconsin (Diagnostic) dataset, evaluate each with
Accuracy, AUC, Precision, Recall, F1, and MCC, and persist the trained models
plus a held-out test CSV for the Streamlit app.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df.rename(columns={"target": "diagnosis"}, inplace=True)
feature_cols = list(data.feature_names)

print(f"Dataset shape: {df.shape} | features: {len(feature_cols)}")

X = df[feature_cols]
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.joblib"))

test_out = X_test.copy()
test_out["diagnosis"] = y_test.values
test_out.to_csv(os.path.join(BASE_DIR, "test_data.csv"), index=False)
print(f"Saved test_data.csv with {test_out.shape[0]} rows.")

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results[name] = metrics

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

results_df = pd.DataFrame(results).T
results_df = results_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
results_df.to_csv(os.path.join(BASE_DIR, "model_comparison.csv"))

with open(os.path.join(BASE_DIR, "model_comparison.json"), "w") as f:
    json.dump(results_df.round(4).to_dict(orient="index"), f, indent=2)

print("\n=== Comparison Table ===")
print(results_df.round(4).to_string())

best_model = results_df["F1"].idxmax()
print(f"\nBest model by F1 score: {best_model}")
