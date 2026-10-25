import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "diabetes.csv")
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.joblib")
PNG_PATH = os.path.join(BASE_DIR, "confusion_matrix.png")
PDF_PATH = os.path.join(BASE_DIR, "confusion_matrix.pdf")
JSON_PATH = os.path.join(BASE_DIR, "confusion_matrix_results.json")

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

def generate_confusion_matrix():
    # 1. Verify and load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run train_model.py first.")
    
    print(f"Loading trained model pipeline from: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)

    # 2. Verify and load dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}.")

    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    X = df[EXPECTED_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    # 3. Apply exact preprocessing as train_model.py:
    # Replace medically implausible zero values with NaN
    implausible_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in implausible_zero_cols:
        X[col] = X[col].replace(0, np.nan)

    # 4. Stratified 80/20 train-test split with random_state=42 (identical to training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Held-out test set size: {len(X_test)} samples (Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()})")

    # 5. Predict on held-out test dataset
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # 6. Calculate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Verify sum equals test samples
    total_test = len(y_test)
    assert (tn + fp + fn + tp) == total_test, "Sum of matrix elements does not match test sample size!"

    # Calculate classification metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))

    print("\n--- Confusion Matrix Counts ---")
    print(f"True Negatives  (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives  (TP): {tp}")
    print(f"Total Test Samples  : {total_test}")
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall   : {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score : {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC  : {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print("--------------------------\n")

    # 7. Plot Confusion Matrix Diagram
    # Set high-resolution professional styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

    fig, ax = plt.subplots(figsize=(7.5, 6.8), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    # Elegant teal/blue sequential colormap
    cmap = plt.cm.YlGnBu

    # Draw heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # Add Colorbar with specific label
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    cbar.set_label('Number of Samples', fontsize=11, fontweight='semibold', labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    # Tick labels
    classes_x = ['Predicted Negative (0)', 'Predicted Positive (1)']
    classes_y = ['Actual Negative (0)', 'Actual Positive (1)']

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes_x, fontsize=10.5, fontweight='semibold', color='#1e293b')
    ax.set_yticklabels(classes_y, fontsize=10.5, fontweight='semibold', color='#1e293b')

    # Axis Labels and Title
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold', labelpad=10, color='#0f172a')
    ax.set_ylabel('Actual Class', fontsize=12, fontweight='bold', labelpad=10, color='#0f172a')
    ax.set_title('Confusion Matrix — DiaPredict Diabetes Prediction System',
                 fontsize=13, fontweight='bold', pad=18, color='#0f172a')

    # Subtle grid lines between cells
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which='minor', color='#ffffff', linestyle='-', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(which='major', length=0)

    # Cell annotation details (TN, FP, FN, TP descriptors + integer count + percentage)
    cell_metadata = [
        [(tn, "True Negative (TN)", tn / total_test), (fp, "False Positive (FP)", fp / total_test)],
        [(fn, "False Negative (FN)", fn / total_test), (tp, "True Positive (TP)", tp / total_test)]
    ]

    threshold = cm.max() / 2.0

    for i in range(2):
        for j in range(2):
            val, label, pct = cell_metadata[i][j]
            # Choose text color based on cell luminance
            text_color = '#ffffff' if cm[i, j] > threshold else '#0f172a'
            sub_color = '#e2e8f0' if cm[i, j] > threshold else '#475569'

            # Primary count
            ax.text(j, i - 0.12, f"{val}",
                    ha="center", va="center",
                    color=text_color, fontsize=20, fontweight='bold')
            # Descriptor
            ax.text(j, i + 0.08, label,
                    ha="center", va="center",
                    color=sub_color, fontsize=10, fontweight='semibold')
            # Percentage of test set
            ax.text(j, i + 0.22, f"({pct*100:.1f}% of test set)",
                    ha="center", va="center",
                    color=sub_color, fontsize=9, style='italic')

    # Summary classification box at the bottom
    summary_text = (
        f"Evaluation Metrics (N={total_test}):\n"
        f"Accuracy: {accuracy*100:.2f}%  |  Precision: {precision*100:.2f}%  |  "
        f"Recall: {recall*100:.2f}%  |  F1-Score: {f1*100:.2f}%  |  ROC-AUC: {roc_auc*100:.2f}%\n"
        f"Counts: TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}"
    )
    plt.figtext(
        0.5, 0.015, summary_text,
        ha='center', fontsize=9.5,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1),
        color='#1e293b'
    )

    plt.tight_layout(rect=[0, 0.07, 1, 0.98])

    # Save PNG
    plt.savefig(PNG_PATH, dpi=300, bbox_inches='tight', facecolor='#ffffff')
    print(f"Confusion Matrix PNG saved to: {PNG_PATH}")

    # Save PDF
    plt.savefig(PDF_PATH, bbox_inches='tight', facecolor='#ffffff')
    print(f"Confusion Matrix PDF saved to: {PDF_PATH}")

    plt.close()

    # 8. Save structured results JSON
    results = {
        "dataset": "Pima Indians Diabetes Dataset",
        "model_file": "backend/diabetes_model.joblib",
        "split": {
            "test_size": 0.2,
            "random_state": 42,
            "stratified": True,
            "total_samples": int(len(df)),
            "train_samples": int(len(X_train)),
            "test_samples": int(total_test)
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "sum_check_passed": bool((tn + fp + fn + tp) == total_test)
        },
        "classification_metrics": {
            "accuracy": round(accuracy, 4),
            "accuracy_percentage": round(accuracy * 100, 2),
            "precision": round(precision, 4),
            "precision_percentage": round(precision * 100, 2),
            "recall": round(recall, 4),
            "recall_percentage": round(recall * 100, 2),
            "f1_score": round(f1, 4),
            "f1_score_percentage": round(f1 * 100, 2),
            "roc_auc": round(roc_auc, 4),
            "roc_auc_percentage": round(roc_auc * 100, 2)
        },
        "class_labels": {
            "0": "Diabetes Negative",
            "1": "Diabetes Positive"
        }
    }

    with open(JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Confusion matrix numeric results saved to: {JSON_PATH}")

    return results

if __name__ == "__main__":
    generate_confusion_matrix()
