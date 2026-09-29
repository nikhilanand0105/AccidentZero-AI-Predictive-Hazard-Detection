## AccidentZero AI – Intelligent Safety Risk Monitoring

AccidentZero AI is an end‑to‑end **industrial safety risk monitoring system** that:

- **Ingests** historical safety data (CSV / Excel, including NULLs)
- **Validates and preprocesses** it with a reusable pipeline
- **Engineers domain‑specific safety features**
- **Trains five complementary models** to quantify accident risk
- **Exposes a FastAPI backend** for real‑time and batch scoring
- **Provides a modern dashboard frontend** with interactive visualizations

This README explains:

- The **overall architecture & flow**
- Each **pipeline stage** and where to find it in the code
- The **five models** used, why they were chosen, and how they compare
- How the **API and frontend** work together
- Suggested **responsibilities for a 5‑person team**

---

## 1. High‑Level Architecture

**Tech stack**

- **Backend / ML**: Python, scikit‑learn, XGBoost, LightGBM, CatBoost, IsolationForest, TensorFlow/Keras
- **API**: FastAPI + Uvicorn
- **Frontend**: Plain HTML/CSS/JS + Chart.js (interactive charts)
- **Storage / Artifacts**: CSV/XLSX data, joblib `.pkl` model files, Keras `.keras` file, preprocessing artifacts

**Key directories**

- `data/` – raw and generated datasets, batch prediction outputs, sample Excel
- `pipeline/` – data loading, validation, preprocessing, feature engineering, prediction helpers
- `models/` – model training code and saved models
- `evaluation/` – EDA, model evaluation metrics
- `api/` – FastAPI application (`app.py`)
- `frontend/` – dashboard UI (`index.html`, `style.css`, `script.js`)
- `utils/` – dataset generation, Excel helpers

---

## 2. End‑to‑End Flow

**Training flow** (run once or when retraining):

1. **Load raw data**
   - File: `pipeline/data_loader.py`
   - Function: `load_data("data/safety_data.csv")`
   - Called from: `main.py`

2. **Data validation**
   - File: `pipeline/data_validator.py`
   - Function: `validate_data(df_raw)`
   - Checks missing values, duplicates, basic sanity; summary printed in `main.py`.

3. **Preprocessing (reusable pipeline)**
   - File: `pipeline/preprocessing.py`
   - Function: `preprocess_data(df_raw.copy(), fit=True)`
   - Steps:
     - `handle_missing_values` – numeric imputation with column means
     - `encode_categorical` – label encodes categorical/text columns
     - `scale_features` – standardizes numeric features with `StandardScaler`
   - Returns:
     - `df_processed` – cleaned & scaled DataFrame
     - `PreprocessArtifacts` – dataclass containing:
       - Column means
       - Label encoders
       - Fitted scaler
       - List of scaled feature columns
   - Artifacts saved to:
     - `models/preprocess_artifacts.pkl`

4. **Feature engineering**
   - File: `pipeline/feature_engineering.py`
   - Function: `engineer_features(df_processed.copy())`
   - Adds derived features:
     - `fatigue_index = shift_hours + overtime_hours`
     - `equipment_risk = equipment_age / (inspection_score + 1)`
     - `weather_severity = temperature * humidity`
   - Result: `df_engineered`

5. **Train/test split**
   - File: `models/train_models.py`
   - Function: `split_data(df_engineered)`
   - Returns: `X_train_fe, X_test_fe, y_train_fe, y_test_fe`
   - Target variable: `accident`

6. **Train five models**
   - File: `models/train_models.py` & `models/lstm_model.py`
   - Functions (all called from `main.py`):
     - `train_xgboost(X_train_fe, y_train_fe)`
     - `train_lightgbm(X_train_fe, y_train_fe)`
     - `train_catboost(X_train_fe, y_train_fe)`
     - `train_isolation_forest(X_train_fe)`
     - `train_lstm(X_train_fe, y_train_fe)`
   - Saved models:
     - `models/xgb.pkl`
     - `models/lgbm.pkl`
     - `models/cat.pkl`
     - `models/iso.pkl`
     - `models/lstm.keras`
   - Feature column schema saved as:
     - `models/feature_columns.json`

7. **Model evaluation**
   - File: `evaluation/evaluate_models.py`
   - Functions:
     - `evaluate_classification_model` – accuracy, precision, recall, F1 (tree models)
     - `evaluate_lstm_model` – same metrics for LSTM
   - Called in `main.py` for:
     - Base vs feature‑engineered versions
     - Comparison printed to console.

8. **Batch prediction on training set (for reports)**
   - File: `pipeline/predict_engine.py`
   - Functions:
     - `load_models()` – loads tree models & IsolationForest
     - `batch_predict(df_engineered, models)` – per‑row ensemble probability and risk classification
   - Output:
     - `data/batch_predictions.csv`

**Inference / API flow** (real‑time + batch):

1. **FastAPI app startup**
   - File: `api/app.py`
   - Loads:
     - Saved models from `models/*.pkl` and `models/lstm.keras`
     - Preprocessing artifacts (`preprocess_artifacts.pkl`)
     - Feature column list (`feature_columns.json`)

2. **Input preparation for inference**
   - Helper in `api/app.py`: `_prepare_features(df_in: pd.DataFrame)`
   - Steps:
     - Normalize/canonicalize NULLs and text via `_normalize_numeric_inputs`
     - Reuse training preprocessing via `preprocess_data(df, artifacts=..., fit=False)`
     - Apply `engineer_features`
     - Align columns to training feature set (add missing with zeros)
     - Replace `inf` / `NaN` with safe defaults

3. **Prediction across all models**
   - Helper in `api/app.py`: `_predict_from_features(X)`
   - For each row:
     - `xgb_probability` – probability from XGBoost
     - `lgbm_probability` – probability from LightGBM
     - `cat_probability` – probability from CatBoost
     - `lstm_probability` – probability from LSTM (sequence representation of features)
     - `iso_anomaly_score` – normalized anomaly score from IsolationForest
     - `ensemble_tree_probability` – weighted average of tree models (XGB, LGBM, CAT)
     - `ensemble_probability` – overall average of all five signals
     - `risk_score` – `ensemble_probability * 100`
     - `risk_level` – LOW / MODERATE / HIGH / CRITICAL

4. **API endpoints**
   - File: `api/app.py`
   - `GET /`
     - Health check: `{ "message": "AccidentZero AI Backend Running" }`
   - `POST /predict`
     - Body: single JSON object with safety parameters
     - Returns: one row of model outputs + `risk_score` + `risk_level`
   - `POST /predict/batch`
     - Body: list of JSON objects
     - Returns:
       - `rows`: per‑row outputs from all five models + ensembles
       - `summary`: averages of numeric outputs across rows
       - `count`: number of rows scored
   - `POST /predict/excel`
     - Body: multipart file (`.xlsx`, multi‑sheet allowed)
     - Reads all sheets, drops empty/repeated headers, handles NULLs
     - Returns same structure as `/predict/batch`

5. **Frontend dashboard**
   - File: `frontend/index.html`, `frontend/style.css`, `frontend/script.js`
   - Features:
     - Manual single‑scenario input form
     - Excel upload for batch predictions
     - Metrics:
       - Accident probability, risk score, risk level
       - Batch summary (rows, average ensemble, average risk score)
     - Interactive charts (Chart.js):
       - **Model Ensemble Overview** – bar chart of average XGB/LGBM/CAT/LSTM/ISO/Ensemble
       - **Risk Distribution (Batch)** – bar chart of row counts per risk band
       - **Selected Row – Model Comparison** – radar chart of per‑model outputs for a clicked row
     - Results table:
       - Shows XGB/LGBM/CAT/LSTM/ISO/Ensemble probabilities + risk for each row
       - Clicking a row updates the radar chart for that row

---

## 3. Block Diagram of the System

You can paste this **Mermaid diagram** into tools that support it (or keep it as documentation):

```mermaid
flowchart LR
    subgraph Data
        A[Raw CSV / Excel\n data/safety_data.csv,\n uploaded .xlsx]
    end

    subgraph Pipeline
        B[Data Loader\npipeline/data_loader.py]
        C[Validation\npipeline/data_validator.py]
        D[Preprocessing\nhandle_missing_values,\nencode_categorical,\nscale_features]
        E[Feature Engineering\npipeline/feature_engineering.py]
    end

    subgraph Training
        F1[XGBoost\nmodels/train_models.py]
        F2[LightGBM\nmodels/train_models.py]
        F3[CatBoost\nmodels/train_models.py]
        F4[IsolationForest\nmodels/train_models.py]
        F5[LSTM\nmodels/lstm_model.py]
        G[Save Models &\nPreprocess Artifacts\nmodels/*.pkl,\nmodels/lstm.keras,\nmodels/preprocess_artifacts.pkl,\nmodels/feature_columns.json]
    end

    subgraph API
        H[FastAPI app\napi/app.py]
        I[_prepare_features()\n+ _predict_from_features()]
    end

    subgraph Frontend
        J[Dashboard UI\nfrontend/index.html]
        K[Charts & Table\nfrontend/script.js\n(Chart.js)]
    end

    A --> B --> C --> D --> E --> F1
    E --> F2
    E --> F3
    E --> F4
    E --> F5

    F1 --> G
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G

    G --> H --> I

    I --> J
    I --> K
```

---

## 4. The Five Models – Roles, Code Locations, and Rationale

### 4.1 XGBoost (eXtreme Gradient Boosting)

- **Type**: Gradient boosted decision trees (supervised, classification)
- **Code**
  - Definition & training: `models/train_models.py` → `train_xgboost`
  - Called from: `main.py`
  - Saved as: `models/xgb.pkl`
  - Used for inference in: `api/app.py` (loaded and called in `_predict_from_features`)
- **Why it’s used**
  - Handles **nonlinear interactions** and tabular numeric data very well.
  - Robust to different feature scales (and we still standardize for stability).
  - Excellent performance on many Kaggle / industrial benchmarks.
- **Why it’s better than many alternatives here**
  - Compared to **plain logistic regression**: captures complex feature interactions (e.g., combinations of fatigue, environment, equipment).
  - Compared to **single decision tree**: less overfitting, better generalization via boosting.

### 4.2 LightGBM

- **Type**: Gradient boosting using histogram‑based, leaf‑wise growth
- **Code**
  - Definition & training: `models/train_models.py` → `train_lightgbm`
  - Called from: `main.py`
  - Saved as: `models/lgbm.pkl`
  - Used for inference in: `api/app.py` (`_predict_from_features`)
- **Why it’s used**
  - Very fast training and inference on CPU.
  - Good at handling **large feature spaces** and sparse inputs.
  - Complementary to XGBoost due to different tree growth strategy.
- **Why it’s better than many alternatives here**
  - Compared to **RandomForest**:
    - Boosting (LightGBM) typically achieves higher accuracy with fewer trees.
  - Compared to using **only XGBoost**:
    - Provides **model diversity** in the ensemble; helps reduce systematic errors.

### 4.3 CatBoost

- **Type**: Gradient boosting with strong support for categorical features
- **Code**
  - Definition & training: `models/train_models.py` → `train_catboost`
  - Called from: `main.py`
  - Saved as: `models/cat.pkl`
  - Used for inference in: `api/app.py` (`_predict_from_features`)
- **Why it’s used**
  - Excellent handling of **categorical variables** and robust default hyperparameters.
  - Often performs strongly with minimal tuning.
- **Why it’s better than many alternatives here**
  - Compared to **Naive Bayes or k‑NN**:
    - CatBoost tends to outperform on structured tabular data.
  - Compared to just using **XGBoost/LightGBM**:
    - Different boosting implementation increases ensemble robustness, especially if more categorical fields are added later (e.g., shift type, site, job role).

### 4.4 IsolationForest (Unsupervised Anomaly Detection)

- **Type**: Unsupervised anomaly detector (tree‑based, isolates outliers)
- **Code**
  - Definition & training: `models/train_models.py` → `train_isolation_forest`
  - Called from: `main.py`
  - Saved as: `models/iso.pkl`
  - Used for inference in: `api/app.py` (via `iso.decision_function` inside `_predict_from_features`)
- **Why it’s used**
  - Captures **rare, abnormal patterns** in the feature space even when labels might not fully describe risk.
  - Adds a **different perspective**: “how unusual is this configuration?”.
- **Why it’s better than many alternatives here**
  - Compared to **One‑Class SVM**:
    - IsolationForest scales better on moderate‑sized tabular data and is easier to tune.
  - Compared to having **no unsupervised model**:
    - Provides an additional signal that can highlight out‑of‑distribution conditions (e.g., extreme fatigue + extreme weather) that may not appear frequently in training labels.

### 4.5 LSTM (Long Short‑Term Memory Network)

- **Type**: Recurrent neural network (sequence modeling)
- **Code**
  - Definition & training: `models/lstm_model.py` → `train_lstm`
  - Called from: `main.py`
  - Saved as: `models/lstm.keras`
  - Used for inference in: `api/app.py`:
    - Features reshaped to `[samples, time_steps=1, features]`
    - Probabilities from `lstm.predict` added as `lstm_probability`
- **Why it’s used**
  - Adds **deep learning** capability that can capture patterns in the transformed feature sequence, even though we currently treat each row as a single time step.
  - Provides a different inductive bias than trees.
- **Why it’s better than many alternatives here**
  - Compared to a **shallow neural network (MLP)**:
    - LSTM structure can be extended later to multi‑step sequences (e.g., history of shifts for a worker).
  - Compared to relying only on **tree ensembles**:
    - Neural networks may capture subtler nonlinearities when scaled up with more features and history.

---

## 5. Ensemble Strategy and Advantage

The system combines the strengths of all models:

- **Tree ensemble (XGBoost + LightGBM + CatBoost)**
  - `ensemble_tree_probability = 0.4 * xgb + 0.3 * lgbm + 0.3 * cat`
  - Optimized for **tabular predictive performance**.

- **Full ensemble (all eight models)**
  - `ensemble_probability` is the mean of:
    - XGBoost, LightGBM, CatBoost, HistGBM, ExtraTrees
    - LSTM, IsolationForest score, Stacking meta‑model
  - Adds:
    - **LSTM** for deep learning patterns.
    - **IsolationForest** for anomaly‑based risk.
    - **HistGBM & ExtraTrees** for additional tree diversity.
    - **StackingClassifier** to learn an optimal combination of tree models.

Because each model has different biases and strengths, the ensemble:

- **Reduces variance** vs a single model.
- **Increases robustness** to distribution shifts.
- **Highlights high‑risk cases** even when a single model is uncertain.

Compared to using **only** one model (e.g., just XGBoost or just logistic regression), this architecture:

- Offers **richer explanations** (per‑model outputs shown in the UI).
- Gives **multiple perspectives** on risk (predictive vs anomalous vs deep patterns).

---

## 6. Suggested Group Responsibilities (5 People)

You can present this as a 5‑member project with clear ownership:

- **Person 1 – Data & Preprocessing Lead**
  - Files:
    - `pipeline/data_loader.py`
    - `pipeline/data_validator.py`
    - `pipeline/preprocessing.py`
    - `utils/generate_dataset.py`
  - Responsibilities:
    - Define data schema and quality checks.
    - Implement missing value handling, encoding, scaling.
    - Manage reusable preprocessing artifacts for inference.

- **Person 2 – Feature Engineering & XGBoost**
  - Files:
    - `pipeline/feature_engineering.py`
    - `models/train_models.py` (XGBoost part)
  - Responsibilities:
    - Design domain‑specific features (fatigue, equipment risk, weather severity).
    - Train and tune **XGBoost** baseline.
    - Compare XGBoost vs simpler baselines (e.g., logistic regression) and justify improvements.

- **Person 3 – Gradient Boosting Suite (LightGBM, CatBoost, HistGBM, ExtraTrees)**
  - Files:
    - `models/train_models.py` (LightGBM, CatBoost, HistGBM, ExtraTrees parts)
    - `evaluation/evaluate_models.py`
  - Responsibilities:
    - Train **LightGBM**, **CatBoost**, **HistGradientBoosting**, and **ExtraTrees**.
    - Benchmark them against XGBoost.
    - Explain why boosting and randomized trees beat simpler models (e.g., decision trees, random forests) on this tabular problem.

- **Person 4 – Anomaly Detection, Deep Learning & Meta‑Ensemble**
  - Files:
    - `models/train_models.py` (IsolationForest + Stacking parts)
    - `models/lstm_model.py`
    - `api/app.py` (`_predict_from_features`, LSTM, ISO & stacking integration)
  - Responsibilities:
    - Implement **IsolationForest** for anomaly‑based risk.
    - Implement and train **LSTM** model.
    - Build and train the **StackingClassifier** meta‑ensemble.
    - Integrate these into the ensemble and describe how they complement tree models.

- **Person 5 – API & Frontend Visualization**
  - Files:
    - `api/app.py`
    - `frontend/index.html`
    - `frontend/style.css`
    - `frontend/script.js`
  - Responsibilities:
    - Build FastAPI endpoints for single, batch JSON, and Excel predictions.
    - Design **modern dashboard UI**, Excel upload flow, and **interactive charts** (Chart.js).
    - Present per‑model outputs and ensemble comparison clearly for end users.

---

## 7. How to Run the Project

From PowerShell:

```powershell
cd "C:\Users\mruni\Desktop\AccidentZeroAI\AccidentZeroAI"

# 1) Create venv & install dependencies
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt

# 2) Train models and generate artifacts (runs full pipeline + EDA)
.\.venv\Scripts\python main.py

# 3) Start FastAPI backend
.\.venv\Scripts\python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Serve frontend (optional but recommended):

```powershell
cd "C:\Users\mruni\Desktop\AccidentZeroAI\AccidentZeroAI\frontend"
..\.\.venv\Scripts\python -m http.server 5500
```

Then open the dashboard:

- `http://127.0.0.1:5500/index.html`

---

## 8. Using This Content in a Word Report

If you need a **Word (.docx) report**:

- Create a new Word document.
- Copy the sections of this README (architecture, block diagram text, model explanations, group roles).
- Paste them into Word and adjust formatting (headings, bullet points, images of charts or the Mermaid diagram if you render it).

This README is structured so it can be directly used as the basis for your written project report.

