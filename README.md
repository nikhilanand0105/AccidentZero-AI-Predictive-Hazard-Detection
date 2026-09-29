# AccidentZero AI - Predictive Hazard Detection

AccidentZero AI is a highly sophisticated, AI-driven hazard tracking architecture engineered exclusively for the preemptive recognition and mitigation of occupational hazards. Moving beyond traditional post-incident response frameworks, this platform leverages advanced computational algorithms to forecast possible workplace mishaps long before they actualize. 

This solution is designed for demanding industrial sectors, including heavy engineering, active construction zones, and large-scale manufacturing facilities.

## Key Features

- **Multi-Model Predictive Engine**: Utilizes a highly diverse coalition of models (CatBoost, LightGBM, XGBoost, HistGradientBoosting, Extra Trees, LSTM, and Isolation Forest) aggregated via a Stacking Classifier to accurately calculate the likelihood of workplace incidents.
- **Risk Categorization**: Systematically compartmentalizes predictions into intuitive risk tiers (Critical, High, Moderate, Low) for rapid comprehension by on-site safety officers.
- **Real-time API Pipeline**: Built on a high-velocity **FastAPI** backend to seamlessly manage digital communication routing, enabling uninterrupted data ingestion and immediate, on-the-fly prognostications.
- **Advanced Feature Engineering**: Synthesizes novel metrics (such as workforce exhaustion from shift and overtime hours, or hardware threat metrics from equipment age and maintenance scores) to vastly improve algorithmic precision.
- **Interactive Dashboard**: A highly responsive, visual command center (Frontend) to translate complex statistical outputs into easily digestible graphics, heatmaps, and tables.
- **Tableau Integration**: Built-in support and exports for advanced visual profiling and enterprise analytics via Tableau.

## Project Structure

- `Project/AccidentZeroAI/`: The core application and machine learning codebase.
  - `api/`: FastAPI backend implementation (`app.py`, `insights.py`).
  - `models/`: Machine learning models and the ensemble engine.
  - `pipeline/`: Data preprocessing, validation, and feature engineering pipelines.
  - `frontend/`: Web-based visual dashboard (HTML, CSS, JS).
  - `evaluation/`: Scripts for Exploratory Data Analysis (EDA) and model evaluation.
  - `data/`: Datasets utilized for training and testing.
  - `utils/`: Helper scripts for data generation, explainability, and fusion extensions.
  - `tableau_exports/`: Processed data ready for Tableau integration.
- `Project_Poster/`: High-resolution project poster.
- `Project_Report/`: In-depth project documentation and academic report.
- `Project_ppt/`: Presentation slides covering the initiative.
- `Visual Output/`: High-quality screenshots of the dashboard and graphical interfaces.

## Data Methodology

The models are trained on highly specialized occupational datasets comprising critical causal variables:
- **Shift Hours & Overtime Hours** (Worker exhaustion metrics)
- **Worker Experience** 
- **Equipment Age & Maintenance Score** (Hardware threat metrics)
- **Temperature & Humidity** (Environmental hostility)
- **Inspection Score**

By harmonizing these distinct variables into a single analytical view, the multi-model architecture achieves a formidable **84% accuracy benchmark** in predicting impending catastrophes, providing safety officers with vital lead time to execute preventive interventions.

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:nikhilanand0105/AccidentZero-AI---Predictive-Hazard-Detection-.git
   cd AccidentZero-AI---Predictive-Hazard-Detection-
   ```
2. **Install Dependencies**:
   (Ensure you have Python installed, then install from the requirements file)
   ```bash
   pip install -r Project/AccidentZeroAI/requirements.txt
   ```
3. **Run the API Server**:
   ```bash
   cd Project/AccidentZeroAI/api
   uvicorn app:app --reload
   ```
4. **Access the Dashboard**:
   Open the `Project/AccidentZeroAI/frontend/index.html` file in your preferred browser, or serve it via a local web server to interact with the API.

## Future Scope

The foundational architecture is primed for massive future expansion, including:
- Direct synchronization with live **IoT sensors** for automated data ingestion.
- The development of a secure mobile interface for on-site laborers.
- Enhanced Explainable AI (XAI) protocols to generate highly detailed, human-readable rationales justifying every hazard warning.
