import React, { useState, useEffect } from "react";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";

const INITIAL_FORM = {
  Pregnancies: "",
  Glucose: "",
  BloodPressure: "",
  SkinThickness: "",
  Insulin: "",
  BMI: "",
  DiabetesPedigreeFunction: "",
  Age: ""
};

const FIELD_METADATA = [
  { key: "Pregnancies", label: "Pregnancies", unit: "count", placeholder: "e.g. 2", min: 0, step: "1" },
  { key: "Glucose", label: "Glucose Level", unit: "mg/dL", placeholder: "e.g. 120", min: 0, step: "1" },
  { key: "BloodPressure", label: "Blood Pressure", unit: "mm Hg", placeholder: "e.g. 70", min: 0, step: "1" },
  { key: "SkinThickness", label: "Skin Thickness", unit: "mm", placeholder: "e.g. 20", min: 0, step: "1" },
  { key: "Insulin", label: "2-Hour Serum Insulin", unit: "mu U/ml", placeholder: "e.g. 80", min: 0, step: "1" },
  { key: "BMI", label: "Body Mass Index (BMI)", unit: "weight in kg/(height in m)^2", placeholder: "e.g. 28.5", min: 0, step: "0.1" },
  { key: "DiabetesPedigreeFunction", label: "Diabetes Pedigree Function", unit: "score", placeholder: "e.g. 0.35", min: 0, step: "any" },
  { key: "Age", label: "Age", unit: "years", placeholder: "e.g. 33", min: 0, step: "1" }
];

export default function App() {
  const [formData, setFormData] = useState(INITIAL_FORM);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [metricsLoading, setMetricsLoading] = useState(true);

  // Fetch metrics on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/metrics`)
      .then((res) => {
        if (!res.ok) throw new Error("Metrics not available yet");
        return res.json();
      })
      .then((data) => {
        setMetrics(data);
        setMetricsLoading(false);
      })
      .catch((err) => {
        console.warn("Could not fetch metrics:", err.message);
        setMetricsLoading(false);
      });
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  const handleReset = () => {
    setFormData(INITIAL_FORM);
    setPrediction(null);
    setErrorMsg("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg("");

    // Validate all 8 fields are filled
    for (const field of FIELD_METADATA) {
      if (formData[field.key] === "" || formData[field.key] === null) {
        setErrorMsg(`Please provide a value for ${field.label}.`);
        return;
      }
      const numVal = parseFloat(formData[field.key]);
      if (isNaN(numVal) || numVal < 0) {
        setErrorMsg(`Please enter a valid non-negative number for ${field.label}.`);
        return;
      }
    }

    setLoading(true);

    try {
      const payload = {
        Pregnancies: parseFloat(formData.Pregnancies),
        Glucose: parseFloat(formData.Glucose),
        BloodPressure: parseFloat(formData.BloodPressure),
        SkinThickness: parseFloat(formData.SkinThickness),
        Insulin: parseFloat(formData.Insulin),
        BMI: parseFloat(formData.BMI),
        DiabetesPedigreeFunction: parseFloat(formData.DiabetesPedigreeFunction),
        Age: parseFloat(formData.Age)
      };

      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction request failed.");
      }

      setPrediction(data);
    } catch (err) {
      setErrorMsg(err.message || "Unable to reach the prediction server. Ensure Flask is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="medical-icon">+</div>
          <div className="header-titles">
            <h1>DiaPredict</h1>
            <p>AI-Powered Diabetes Prediction</p>
          </div>
        </div>
        <div className="header-badge">ML MINI PROJECT</div>
      </header>

      {/* Main Hero Card */}
      <div className="intro-card">
        <h2>Diabetes Prediction System</h2>
        <p>
          This application demonstrates a machine learning classifier (Logistic Regression) trained on the
          Pima Indians Diabetes Dataset to evaluate risk estimation using clinical indicators.
        </p>
      </div>

      {/* Main Grid: Form & Output */}
      <div className="main-grid">
        {/* Left: Input Form */}
        <div className="form-card">
          <h3 className="section-title">Patient Diagnostic Parameters</h3>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              {FIELD_METADATA.map((f) => (
                <div key={f.key} className="input-group">
                  <label htmlFor={f.key}>
                    {f.label} <span className="unit">({f.unit})</span>
                  </label>
                  <input
                    id={f.key}
                    name={f.key}
                    type="number"
                    step={f.step}
                    min={f.min}
                    placeholder={f.placeholder}
                    value={formData[f.key]}
                    onChange={handleChange}
                    className="input-field"
                    required
                  />
                </div>
              ))}
            </div>

            {errorMsg && <div className="error-banner">{errorMsg}</div>}

            <div className="form-actions">
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? "Analyzing Parameters..." : "Predict Diabetes Risk"}
              </button>
              <button type="button" onClick={handleReset} className="btn-secondary" disabled={loading}>
                Reset
              </button>
            </div>
          </form>
        </div>

        {/* Right: Prediction Panel */}
        <div className="result-card">
          <h3 className="section-title">Prediction Result</h3>

          {!prediction ? (
            <div className="empty-state">
              <div className="empty-state-icon">📋</div>
              <strong>No prediction yet</strong>
              <p style={{ marginTop: "4px", fontSize: "0.85rem" }}>
                Fill in the 8 clinical indicators and click "Predict Diabetes Risk" to generate model estimation.
              </p>
            </div>
          ) : (
            <div>
              <div className={`prediction-outcome ${prediction.prediction === 1 ? "positive" : "negative"}`}>
                <div className="outcome-icon">
                  {prediction.prediction === 1 ? "⚠️" : "✅"}
                </div>
                <div className="outcome-details">
                  <h3>{prediction.prediction_label}</h3>
                  <p>Classification Class: {prediction.prediction}</p>
                </div>
              </div>

              <div className="probability-box">
                <div className="probability-header">
                  <span>Model Estimated Probability:</span>
                  <strong>{prediction.probability_percentage}%</strong>
                </div>
                <div className="progress-track">
                  <div
                    className={`progress-fill ${prediction.probability >= 0.5 ? "high" : ""}`}
                    style={{ width: `${Math.min(prediction.probability_percentage, 100)}%` }}
                  />
                </div>
                <p className="prob-note">
                  * Note: This is an empirical machine-learning estimate from logistic regression, not a clinically calibrated risk.
                </p>
              </div>

              <button
                type="button"
                onClick={handleReset}
                className="btn-secondary"
                style={{ width: "100%", marginTop: "12px" }}
              >
                Perform Another Assessment
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Actual Model Metrics Section */}
      <div className="metrics-card">
        <h3 className="section-title">Model Performance & Evaluation Metrics (Held-Out Test Set)</h3>
        {metricsLoading ? (
          <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>Loading model metrics...</p>
        ) : metrics ? (
          <>
            <div className="metrics-grid">
              <div className="metric-item">
                <div className="metric-label">Accuracy</div>
                <div className="metric-val">{(metrics.accuracy * 100).toFixed(1)}%</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">Precision</div>
                <div className="metric-val">{(metrics.precision * 100).toFixed(1)}%</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">Recall</div>
                <div className="metric-val">{(metrics.recall * 100).toFixed(1)}%</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">F1-Score</div>
                <div className="metric-val">{(metrics.f1_score * 100).toFixed(1)}%</div>
              </div>
              <div className="metric-item">
                <div className="metric-label">ROC-AUC</div>
                <div className="metric-val">{(metrics.roc_auc * 100).toFixed(1)}%</div>
              </div>
            </div>
            <div className="metrics-meta">
              Total Dataset: {metrics.dataset_total_samples} samples | Train Set: {metrics.train_samples} | Test Set: {metrics.test_samples}
            </div>
          </>
        ) : (
          <p style={{ color: "var(--danger)", fontSize: "0.85rem" }}>
            Model metrics are not available. Ensure backend is running and model has been trained.
          </p>
        )}
      </div>

      {/* Mandatory Disclaimer */}
      <footer className="disclaimer-card">
        <div className="disclaimer-title">
          <span>ℹ️</span> Educational Medical Disclaimer
        </div>
        <p>
          For educational purposes only. This prototype is not clinically validated and must not be used to
          diagnose diabetes or make treatment decisions.
        </p>
      </footer>
    </div>
  );
}
