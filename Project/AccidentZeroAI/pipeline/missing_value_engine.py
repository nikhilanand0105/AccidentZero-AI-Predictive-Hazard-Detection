"""
KNN-based imputation with per-cell flags (predicted vs original).
Used before visualization and model scoring; does not replace stored raw files.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

# Columns expected by the safety model / dashboard (numeric inputs)
DEFAULT_FEATURE_COLUMNS = [
    "shift_hours",
    "overtime_hours",
    "worker_experience",
    "equipment_age",
    "maintenance_score",
    "temperature",
    "humidity",
    "inspection_score",
]


def impute_numeric_with_knn(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    n_neighbors: int = 5,
    fallback_fill: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Impute missing values in numeric feature columns using KNN (or mean fallback).

    Returns:
        cleaned_df: same columns as input for feature cols; non-feature cols unchanged
        flags_df: boolean DataFrame, True where value was imputed (predicted)
        meta: counts and per-column imputed totals
    """
    feature_cols = feature_cols or DEFAULT_FEATURE_COLUMNS
    df = df.copy()
    present = [c for c in feature_cols if c in df.columns]
    if not present:
        empty_flags = pd.DataFrame(index=df.index)
        return df, empty_flags, {"missing_counts_before": {}, "imputed_cell_counts": {}, "method": "none"}

    fallback_fill = dict(fallback_fill or {})

    # Track missing *before* imputation
    missing_before: dict[str, int] = {}
    for c in present:
        s = pd.to_numeric(df[c], errors="coerce")
        missing_before[c] = int(s.isna().sum())

    # Work on numeric matrix only for selected columns
    X = df[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    missing_mask = np.isnan(X)

    n_rows = X.shape[0]
    n_missing_total = int(missing_mask.sum())

    if n_missing_total == 0:
        flags = pd.DataFrame(False, index=df.index, columns=present)
        return df, flags, {
            "missing_counts_before": missing_before,
            "imputed_cell_counts": {c: 0 for c in present},
            "method": "none",
        }

    # Choose strategy: KNN needs at least 2 rows with some non-NaN overlap
    k = max(1, min(n_neighbors, max(1, n_rows - 1)))

    if n_rows >= 2:
        imputer = KNNImputer(n_neighbors=k, weights="distance")
        try:
            X_filled = imputer.fit_transform(X)
            method = f"knn_k{k}"
        except Exception:
            X_filled = SimpleImputer(strategy="median").fit_transform(X)
            method = "simple_median"
    else:
        # Single row: use fallback means / column medians
        col_medians = np.nanmedian(X, axis=0)
        for j, c in enumerate(present):
            if np.isnan(col_medians[j]):
                col_medians[j] = fallback_fill.get(c, 0.0)
        X_filled = X.copy()
        for i in range(n_rows):
            for j in range(len(present)):
                if np.isnan(X_filled[i, j]):
                    X_filled[i, j] = col_medians[j]
                    if np.isnan(X_filled[i, j]):
                        X_filled[i, j] = fallback_fill.get(present[j], 0.0)
        method = "row_median_fallback"

    imputed_mask = missing_mask.copy()
    # Observed cells: keep original numeric values (no KNN drift on known data)
    for i in range(n_rows):
        for j in range(len(present)):
            if not missing_mask[i, j]:
                X_filled[i, j] = X[i, j]

    # Imputed cells still NaN (e.g. degenerate column): use training fallbacks
    for i in range(n_rows):
        for j in range(len(present)):
            if missing_mask[i, j] and np.isnan(X_filled[i, j]):
                X_filled[i, j] = float(fallback_fill.get(present[j], 0.0))

    for j, c in enumerate(present):
        df[c] = X_filled[:, j]

    flags = pd.DataFrame(imputed_mask, index=df.index, columns=present)
    imputed_counts = {c: int(flags[c].sum()) for c in present}

    return df, flags, {
        "missing_counts_before": missing_before,
        "imputed_cell_counts": imputed_counts,
        "method": method,
        "total_imputed_cells": int(imputed_mask.sum()),
    }


def flags_to_row_dicts(flags_df: pd.DataFrame) -> list[dict[str, bool]]:
    """Serialize boolean flags per row for JSON APIs."""
    if flags_df.empty:
        return []
    out: list[dict[str, bool]] = []
    for _, row in flags_df.iterrows():
        out.append({str(k): bool(v) for k, v in row.items()})
    return out


def merge_flag_columns(df: pd.DataFrame, flags_df: pd.DataFrame, *, suffix: str = "_value_predicted") -> pd.DataFrame:
    """Add explicit columns: <col>_value_predicted True if KNN filled that cell."""
    out = df.copy()
    for c in flags_df.columns:
        col_name = f"{c}{suffix}"
        out[col_name] = flags_df[c].values
    return out
