# DiaPredict - AI-Powered Diabetes Prediction System

An end-to-end Healthcare AI/ML mini-project featuring a Scikit-Learn Logistic Regression model, a Python Flask REST API, and a modern React + Vite frontend.

---

## 📌 Project Overview
- **Aim**: Develop a healthcare tool with a user interface using AI/ML models and algorithms.
- **Model**: Logistic Regression with median imputation (`SimpleImputer`) and feature standardization (`StandardScaler`).
- **Dataset**: Pima Indians Diabetes Dataset (768 records, 8 clinical features).
- **Frontend**: React (Vite) + Plain CSS.
- **Backend**: Python + Flask + Flask-CORS.

---

## 🚀 Running the Application (Windows PowerShell)

Open two separate Windows PowerShell windows:

### Terminal 1: Backend Setup & Server

```powershell
# 1. Navigate to backend directory
cd backend

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the model (downloads dataset and exports artifacts)
python train_model.py

# 6. Start the Flask API
python app.py
```
> The Flask backend will run at `http://127.0.0.1:5000`.

---

### Terminal 2: Frontend Setup & Dev Server

```powershell
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Launch Vite dev server
npm run dev
```
> Open your browser at the URL shown in terminal (typically `http://localhost:5173`).

---

## 🧪 Running Automated Tests

To run the backend test suite:
```powershell
cd backend
.\.venv\Scripts\python test_app.py
```

To verify the frontend production build:
```powershell
cd frontend
npm run build
```

### End-to-End Browser Tests (Playwright)

Install the Playwright test runner and Chromium browser (first time only):
```powershell
cd frontend
npm install
npx playwright install chromium
```

Run all 5 automated End-to-End tests against the real Flask backend & React UI:
```powershell
cd frontend
npm run test:e2e
```
*(Playwright automatically reuses or starts the Vite and Flask development servers)*

View the generated HTML report:
```powershell
cd frontend
npm run test:e2e:report
```

Captured screenshots are automatically saved to:
`frontend/test-results/screenshots/`
- `01-homepage.png`
- `02-successful-prediction.png`
- `03-form-validation.png`
- `04-reset-form.png`
- `05-updated-prediction.png`

---

## 📁 Dataset Details & Manual Download (Fallback)
The training script automatically fetches the dataset from:
`https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv`

If an automated network download fails:
1. Download the CSV directly from the URL above.
2. Place it in `backend/datasets/diabetes.csv`.
3. Run `python train_model.py`.

The 8 diagnostic features:
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (weight in kg/(height in m)^2)
- `DiabetesPedigreeFunction`: Diabetes pedigree function score
- `Age`: Age in years

---

## ⚠️ Medical Disclaimer
For educational purposes only. This prototype is not clinically validated and must not be used to diagnose diabetes or make treatment decisions.
