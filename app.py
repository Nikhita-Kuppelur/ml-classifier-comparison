"""
Streamlit app: upload a test CSV (features + diagnosis target column),
pick a trained classifier, and view its evaluation metrics, confusion
matrix, and classification report. Models are loaded pre-trained from
disk -- no retraining happens on uploaded data.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TARGET_COL = "diagnosis"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")
st.title("Breast Cancer Diagnosis -- Classifier Comparison App")
st.caption(
    "Dataset: UCI Breast Cancer Wisconsin (Diagnostic) | "
    "5 pre-trained classifiers evaluated on uploaded test data"
)


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }
    return scaler, feature_cols, models


scaler, feature_cols, models = load_artifacts()

st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (must include all feature columns + 'diagnosis' target column)",
    type=["csv"],
)

st.sidebar.header("2. Select Model(s)")
model_choice = st.sidebar.selectbox(
    "Choose a model to inspect in detail", list(models.keys())
)
show_all = st.sidebar.checkbox("Compare all 5 models", value=True)

if uploaded_file is None:
    st.info("Upload a test CSV from the sidebar to see results. A sample `test_data.csv` is included in the repo.")
    st.stop()

df = pd.read_csv(uploaded_file)

missing_cols = [c for c in feature_cols if c not in df.columns]
if missing_cols or TARGET_COL not in df.columns:
    st.error(
        f"Uploaded file is missing required columns. "
        f"Missing features: {missing_cols or 'none'}. "
        f"Target column '{TARGET_COL}' present: {TARGET_COL in df.columns}."
    )
    st.stop()

X = df[feature_cols]
y_true = df[TARGET_COL]
X_scaled = scaler.transform(X)

st.subheader("Uploaded Data Preview")
st.dataframe(df.head())


def evaluate(model, X_scaled, y_true):
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return y_pred, metrics


if show_all:
    st.subheader("Metric Comparison -- All 5 Models")
    rows = {}
    preds = {}
    for name, model in models.items():
        y_pred, metrics = evaluate(model, X_scaled, y_true)
        rows[name] = metrics
        preds[name] = y_pred
    comparison_df = pd.DataFrame(rows).T[
        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    st.dataframe(comparison_df.style.format("{:.4f}"))

    best = comparison_df["F1"].idxmax()
    st.success(f"Best performing model on this test data (by F1 score): **{best}**")

st.subheader(f"Detailed Results -- {model_choice}")
model = models[model_choice]
y_pred, metrics = evaluate(model, X_scaled, y_true)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
col1.metric("AUC", f"{metrics['AUC']:.4f}")
col2.metric("Precision", f"{metrics['Precision']:.4f}")
col2.metric("Recall", f"{metrics['Recall']:.4f}")
col3.metric("F1 Score", f"{metrics['F1']:.4f}")
col3.metric("MCC", f"{metrics['MCC']:.4f}")

left, right = st.columns(2)

with left:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification Report**")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"))
