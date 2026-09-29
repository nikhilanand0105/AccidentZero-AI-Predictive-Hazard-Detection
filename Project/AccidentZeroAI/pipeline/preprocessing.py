from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

@dataclass
class PreprocessArtifacts:
    means: dict[str, float]
    encoders: dict[str, LabelEncoder]
    scaler: StandardScaler
    scaled_feature_cols: list[str]


def _safe_label_transform(le: LabelEncoder, values) -> np.ndarray:
    classes = set(le.classes_.tolist())
    mapped = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            mapped.append(None)
        else:
            s = str(v)
            mapped.append(s if s in classes else None)

    out = np.full(len(mapped), -1, dtype=int)
    known_idx = [i for i, v in enumerate(mapped) if v is not None]
    if known_idx:
        out[np.array(known_idx)] = le.transform([mapped[i] for i in known_idx])
    return out


def handle_missing_values(df, *, means: dict[str, float] | None = None, fit: bool = True):
    if fit:
        means = df.mean(numeric_only=True).to_dict()
    means = means or {}
    df = df.copy()
    for col, m in means.items():
        if col in df.columns:
            df[col] = df[col].fillna(m)
    print("[OK] Missing values handled")
    return df, means


def encode_categorical(
    df,
    *,
    encoders: dict[str, LabelEncoder] | None = None,
    fit: bool = True,
):
    df = df.copy()
    encoders = {} if (encoders is None) else dict(encoders)

    for col in df.select_dtypes(include=['object']).columns:
        if fit:
            le = LabelEncoder()
            df[col] = df[col].astype("string")
            le.fit(df[col].fillna("").astype(str))
            encoders[col] = le
            df[col] = _safe_label_transform(le, df[col].tolist())
        else:
            le = encoders.get(col)
            if le is None:
                df[col] = df[col].astype("string").fillna("").astype(str)
                df[col] = df[col].map(lambda s: hash(s) % 10_000_000).astype(int)
            else:
                df[col] = df[col].astype("string")
                df[col] = _safe_label_transform(le, df[col].tolist())
        print(f"[OK] Encoded column: {col}")

    return df, encoders


def scale_features(
    df,
    *,
    scaler: StandardScaler | None = None,
    fit: bool = True,
    scaled_feature_cols: list[str] | None = None,
):
    df = df.copy()

    if scaled_feature_cols is None:
        if "accident" in df.columns:
            feature_cols = df.drop("accident", axis=1).columns.tolist()
        else:
            feature_cols = df.columns.tolist()
    else:
        feature_cols = list(scaled_feature_cols)

    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        if scaler is None:
            raise ValueError("scaler is required when fit=False")
        df[feature_cols] = scaler.transform(df[feature_cols])

    print("[OK] Features scaled")
    return df, scaler, feature_cols


def preprocess_data(df, *, artifacts: PreprocessArtifacts | None = None, fit: bool = True):
    if fit:
        df, means = handle_missing_values(df, fit=True)
        df, encoders = encode_categorical(df, fit=True)
        df, scaler, scaled_cols = scale_features(df, fit=True)
        artifacts = PreprocessArtifacts(
            means=means,
            encoders=encoders,
            scaler=scaler,
            scaled_feature_cols=scaled_cols,
        )
        return df, artifacts

    if artifacts is None:
        raise ValueError("artifacts is required when fit=False")

    df, _ = handle_missing_values(df, means=artifacts.means, fit=False)
    df, _ = encode_categorical(df, encoders=artifacts.encoders, fit=False)
    df, _, _ = scale_features(
        df,
        scaler=artifacts.scaler,
        fit=False,
        scaled_feature_cols=artifacts.scaled_feature_cols,
    )

    return df, artifacts