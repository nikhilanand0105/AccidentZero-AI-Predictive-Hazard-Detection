from pipeline.data_loader import load_data
from pipeline.data_validator import validate_data
from pipeline.preprocessing import preprocess_data
from pipeline.feature_engineering import engineer_features

from evaluation.eda import perform_eda
from evaluation.evaluate_models import *

from models.train_models import *
from models.lstm_model import train_lstm
from models.ensemble_engine import *
from pipeline.predict_engine import *
from utils.explainability_engine import generate_explanation
import json
from pathlib import Path

import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

df_raw = load_data("data/safety_data.csv")

perform_eda(df_raw, title="RAW Dataset EDA")

report = validate_data(df_raw)
print("\n[INFO] Validation Report:")
print(report)

df_processed, preprocess_artifacts = preprocess_data(df_raw.copy(), fit=True)

Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(preprocess_artifacts, "models/preprocess_artifacts.pkl")
print("[OK] Preprocessing artifacts saved -> models/preprocess_artifacts.pkl")

perform_eda(df_processed, title="PREPROCESSED Dataset EDA")

print("\n[INFO] TRAINING WITHOUT FEATURE ENGINEERING")

X_train, X_test, y_train, y_test = split_data(df_processed)

xgb_base = train_xgboost(X_train, y_train)
lgbm_base = train_lightgbm(X_train, y_train)
cat_base = train_catboost(X_train, y_train)

results_base = []
results_base.append(evaluate_classification_model(xgb_base, X_test, y_test, "XGBoost (Base)"))
results_base.append(evaluate_classification_model(lgbm_base, X_test, y_test, "LightGBM (Base)"))
results_base.append(evaluate_classification_model(cat_base, X_test, y_test, "CatBoost (Base)"))

print("\n[INFO] TRAINING WITH FEATURE ENGINEERING")

df_engineered = engineer_features(df_processed.copy())

X_train_fe, X_test_fe, y_train_fe, y_test_fe = split_data(df_engineered)

feature_columns = X_train_fe.columns.tolist()
Path("models").mkdir(parents=True, exist_ok=True)
Path("models/feature_columns.json").write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
print("[OK] Feature columns saved -> models/feature_columns.json")

xgb_fe = train_xgboost(X_train_fe, y_train_fe)
lgbm_fe = train_lightgbm(X_train_fe, y_train_fe)
cat_fe = train_catboost(X_train_fe, y_train_fe)
iso_fe = train_isolation_forest(X_train_fe)
lstm_fe = train_lstm(X_train_fe, y_train_fe)

hgb_fe = train_hist_gradient_boosting(X_train_fe, y_train_fe)
extra_fe = train_extra_trees(X_train_fe, y_train_fe)

print("[INFO] Training Stacking ensemble...")
stack_estimators = [
    ("xgb", xgb_fe),
    ("lgbm", lgbm_fe),
    ("cat", cat_fe),
    ("hgb", hgb_fe),
    ("extra", extra_fe),
]
stack_final = LogisticRegression(max_iter=1000)
stack_clf = StackingClassifier(
    estimators=stack_estimators,
    final_estimator=stack_final,
    stack_method="predict_proba",
    passthrough=False,
)
stack_clf.fit(X_train_fe, y_train_fe)
print("[OK] Stacking ensemble trained")

save_model(xgb_fe, "models/xgb.pkl")
save_model(lgbm_fe, "models/lgbm.pkl")
save_model(cat_fe, "models/cat.pkl")
save_model(iso_fe, "models/iso.pkl")
lstm_fe.save("models/lstm.keras")
print("[OK] Model saved -> models/lstm.keras")

save_model(hgb_fe, "models/hgb.pkl")
save_model(extra_fe, "models/extra_trees.pkl")
save_model(stack_clf, "models/stacking.pkl")
print("\n[OK] ALL ADVANCED MODELS TRAINED & SAVED")

results_fe = []
results_fe.append(evaluate_classification_model(xgb_fe, X_test_fe, y_test_fe, "XGBoost (FE)"))
results_fe.append(evaluate_classification_model(lgbm_fe, X_test_fe, y_test_fe, "LightGBM (FE)"))
results_fe.append(evaluate_classification_model(cat_fe, X_test_fe, y_test_fe, "CatBoost (FE)"))
results_fe.append(evaluate_classification_model(hgb_fe, X_test_fe, y_test_fe, "HistGBM (FE)"))
results_fe.append(evaluate_classification_model(extra_fe, X_test_fe, y_test_fe, "ExtraTrees (FE)"))
results_fe.append(evaluate_classification_model(stack_clf, X_test_fe, y_test_fe, "Stacking (FE)"))
results_fe.append(evaluate_lstm_model(lstm_fe, X_test_fe, y_test_fe))

print("\n[INFO] BASE MODELS vs FEATURE ENGINEERED MODELS")

print("\n[INFO] WITHOUT Feature Engineering:")
for r in results_base:
    print(r)

print("\n[INFO] WITH Feature Engineering:")
for r in results_fe:
    print(r)

sample = X_test_fe.iloc[0:1]

xgb_prob = xgb_fe.predict_proba(sample)[0][1]
lgbm_prob = lgbm_fe.predict_proba(sample)[0][1]
cat_prob = cat_fe.predict_proba(sample)[0][1]

final_prob = ensemble_prediction(xgb_prob, lgbm_prob, cat_prob)

risk_score = compute_risk_score(final_prob)
risk_level = classify_risk_level(risk_score)

print("\n[INFO] FINAL SAFETY INTELLIGENCE OUTPUT")
print("Accident Probability:", round(final_prob, 4))
print("Risk Score:", risk_score)
print("Risk Level:", risk_level)

explanation_row = df_engineered.iloc[0]

reasons = generate_explanation(explanation_row)

print("\n[INFO] DECISION EXPLANATION")
for r in reasons:
    print("•", r)

models = load_models()
batch_results = batch_predict(df_engineered.copy(), models)

batch_results.to_csv("data/batch_predictions.csv", index=False)

print("\n[OK] Batch predictions saved -> data/batch_predictions.csv")