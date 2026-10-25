import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
DATASET_PATH = os.path.join(DATASETS_DIR, "diabetes.csv")
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

EXPECTED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]
TARGET_COLUMN = "Outcome"

def download_dataset():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    if os.path.exists(DATASET_PATH):
        print(f"Dataset already exists at: {DATASET_PATH}")
        return

    print(f"Downloading Pima Indians Diabetes Dataset from {DATA_URL}...")
    try:
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
        with open(DATASET_PATH, "wb") as f:
            f.write(response.content)
        print(f"Dataset downloaded and saved to: {DATASET_PATH}")
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Failed to download dataset automatically.")
        print(f"Details: {e}")
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print(f"1. Download the CSV file directly from:\n   {DATA_URL}")
        print(f"2. Save the file exactly as:\n   {DATASET_PATH}")
        print("3. Re-run this script: python train_model.py")
        print("=" * 60 + "\n")
        sys.exit(1)

def train_and_evaluate():
    download_dataset()

    # Load data
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Failed to read dataset from {DATASET_PATH}: {e}")
        sys.exit(1)

    # Validate columns
    required_cols = EXPECTED_FEATURES + [TARGET_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' missing from {DATASET_PATH}")
            sys.exit(1)

    print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")

    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    # Replace medically implausible zeros with NaN in features
    # Zero in these values represents missing medical readings
    implausible_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in implausible_zero_cols:
        X[col] = X[col].replace(0, np.nan)

    # Stratified train-test split (fit preprocessing strictly on train set to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create preprocessing and classification pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    print("Fitting model pipeline on training set...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model on held-out test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    metrics = {
        "dataset_total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4)
    }

    print("\n--- Test Set Evaluation Metrics ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("-----------------------------------\n")

    # Save fitted model pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
