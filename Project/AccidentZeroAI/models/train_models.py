from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier, ExtraTreesClassifier
import joblib
import numpy as np


def split_data(df):
    X = df.drop("accident", axis=1)
    y = df["accident"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("[OK] Data split completed")
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train):
    print("[INFO] Training XGBoost...")
    model = XGBClassifier()
    model.fit(X_train, y_train)
    print("[OK] XGBoost trained")
    return model


def train_lightgbm(X_train, y_train):
    print("[INFO] Training LightGBM...")
    model = LGBMClassifier()
    model.fit(X_train, y_train)
    print("[OK] LightGBM trained")
    return model


def train_catboost(X_train, y_train):
    print("[INFO] Training CatBoost...")
    model = CatBoostClassifier(verbose=0)
    model.fit(X_train, y_train)
    print("[OK] CatBoost trained")
    return model


def train_hist_gradient_boosting(X_train, y_train):
    print("[INFO] Training HistGradientBoosting...")
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    print("[OK] HistGradientBoosting trained")
    return model


def train_extra_trees(X_train, y_train):
    print("[INFO] Training ExtraTrees...")
    model = ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[OK] ExtraTrees trained")
    return model


def train_isolation_forest(X_train):
    print("[INFO] Training Isolation Forest...")
    model = IsolationForest(contamination=0.05)
    model.fit(X_train)
    print("[OK] Isolation Forest trained")
    return model


def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"[OK] Model saved -> {filename}")