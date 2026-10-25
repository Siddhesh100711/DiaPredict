import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

app = Flask(__name__)
# Enable CORS for frontend requests
CORS(app)

# Global variables loaded on startup
model = None
metrics = None

def load_artifacts():
    global model, metrics

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"[DiaPredict] Loaded model pipeline from {MODEL_PATH}")
        except Exception as e:
            print(f"[DiaPredict] Error loading model: {e}")
            model = None
    else:
        print(f"[DiaPredict] Warning: Model file not found at {MODEL_PATH}. Run train_model.py first.")

    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            print(f"[DiaPredict] Loaded metrics from {METRICS_PATH}")
        except Exception as e:
            print(f"[DiaPredict] Error loading metrics: {e}")
            metrics = None
    else:
        print(f"[DiaPredict] Warning: Metrics file not found at {METRICS_PATH}.")

# Load artifacts when server starts
load_artifacts()

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "DiaPredict Backend API",
        "model_loaded": model is not None,
        "metrics_loaded": metrics is not None
    }), 200

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    if metrics is None:
        return jsonify({
            "error": "Metrics not available. Ensure backend/train_model.py has been executed."
        }), 503
    return jsonify(metrics), 200

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model is not trained or loaded. Please run 'python train_model.py' to generate the model artifact."
        }), 503

    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400

    data = request.get_json()

    # Validate that all expected features are present and numeric
    missing_fields = []
    invalid_fields = []
    parsed_values = {}

    for feature in EXPECTED_FEATURES:
        if feature not in data:
            missing_fields.append(feature)
            continue

        raw_val = data[feature]
        if raw_val is None or raw_val == "":
            missing_fields.append(feature)
            continue

        try:
            val = float(raw_val)
            # Medically sensible non-negativity constraint
            if val < 0:
                invalid_fields.append(f"{feature} cannot be negative")
            else:
                parsed_values[feature] = val
        except (ValueError, TypeError):
            invalid_fields.append(f"{feature} must be a valid number")

    if missing_fields:
        return jsonify({
            "error": "Missing required fields.",
            "missing_fields": missing_fields
        }), 400

    if invalid_fields:
        return jsonify({
            "error": "Invalid field values.",
            "details": invalid_fields
        }), 400

    try:
        # Build DataFrame with features in exact order
        input_df = pd.DataFrame([{feature: parsed_values[feature] for feature in EXPECTED_FEATURES}])

        # Handle zero replacements for imputer matching training logic
        implausible_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in implausible_zero_cols:
            if input_df[col].iloc[0] == 0:
                input_df[col] = np.nan

        # Predict class and probability
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        positive_prob = float(probabilities[1])

        # Formulate response (do not save or log user health data)
        return jsonify({
            "prediction": prediction,
            "prediction_label": "High Risk / Likely Positive" if prediction == 1 else "Low Risk / Likely Negative",
            "probability": round(positive_prob, 4),
            "probability_percentage": round(positive_prob * 100, 2),
            "disclaimer": "For educational purposes only. This prototype is not clinically validated and must not be used to diagnose diabetes or make treatment decisions."
        }), 200

    except Exception as e:
        return jsonify({
            "error": "An error occurred during model inference.",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
