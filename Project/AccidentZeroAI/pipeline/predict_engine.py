import joblib
import pandas as pd
import numpy as np


def load_models():
    print("[INFO] Loading saved models...")
    models = {
        "xgb": joblib.load("models/xgb.pkl"),
        "lgbm": joblib.load("models/lgbm.pkl"),
        "cat": joblib.load("models/cat.pkl"),
        "iso": joblib.load("models/iso.pkl")
    }
    print("[OK] Models loaded successfully")
    return models


def batch_predict(df, models):
    print("\n[INFO] Running Batch Predictions...")
    X = df.drop("accident", axis=1)
    xgb_probs = models["xgb"].predict_proba(X)[:, 1]
    lgbm_probs = models["lgbm"].predict_proba(X)[:, 1]
    cat_probs = models["cat"].predict_proba(X)[:, 1]
    final_probs = (
        0.4 * xgb_probs +
        0.3 * lgbm_probs +
        0.3 * cat_probs
    )
    risk_scores = final_probs * 100
    df["ensemble_probability"] = final_probs
    df["risk_score"] = risk_scores.round(2)
    df["risk_level"] = df["risk_score"].apply(classify_risk_level)
    print("[OK] Batch prediction completed")
    return df


def classify_risk_level(score):
    if score < 30:
        return "LOW"
    elif score < 60:
        return "MODERATE"
    elif score < 80:
        return "HIGH"
    else:
        return "CRITICAL"