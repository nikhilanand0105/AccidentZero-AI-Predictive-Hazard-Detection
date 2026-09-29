import numpy as np

# Full 8-model linear ensemble: P = Σᵢ wᵢ pᵢ, Σwᵢ = 1.
# Tree core (xgb + LightGBM + CatBoost) carries 40% of total mass in the same 40:30:30 ratio as `ensemble_prediction`.
# Remaining 60% is split across HistGB, ExtraTrees, LSTM, IsolationForest, Stacking (see weights below).
FULL_ENSEMBLE_WEIGHTS = np.array(
    [
        0.16,  # xgb   (0.40 * 0.40)
        0.12,  # lgbm  (0.40 * 0.30)
        0.12,  # cat   (0.40 * 0.30)
        0.12,  # hgb
        0.12,  # extra_trees
        0.12,  # lstm
        0.10,  # iso (normalized anomaly score, 0–1)
        0.14,  # stacking
    ],
    dtype=float,
)

FULL_ENSEMBLE_WEIGHT_LABELS = (
    "xgb",
    "lgbm",
    "cat",
    "hgb",
    "extra_trees",
    "lstm",
    "iso",
    "stacking",
)

FULL_ENSEMBLE_WEIGHTS_DICT = dict(zip(FULL_ENSEMBLE_WEIGHT_LABELS, FULL_ENSEMBLE_WEIGHTS.tolist()))


def compute_weighted_ensemble_probability(
    xgb_prob,
    lgbm_prob,
    cat_prob,
    hgb_prob,
    extra_prob,
    lstm_prob,
    iso_score,
    stack_prob,
) -> np.ndarray:
    """Vectorized weighted sum; each argument is 1D array-like of length n_rows."""
    M = np.column_stack(
        [
            np.asarray(xgb_prob, dtype=float),
            np.asarray(lgbm_prob, dtype=float),
            np.asarray(cat_prob, dtype=float),
            np.asarray(hgb_prob, dtype=float),
            np.asarray(extra_prob, dtype=float),
            np.asarray(lstm_prob, dtype=float),
            np.asarray(iso_score, dtype=float),
            np.asarray(stack_prob, dtype=float),
        ]
    )
    return M @ FULL_ENSEMBLE_WEIGHTS


def ensemble_prediction(xgb_prob, lgbm_prob, cat_prob):
    print("\n[INFO] Running Ensemble Intelligence...")
    final_prob = (
        0.4 * xgb_prob +
        0.3 * lgbm_prob +
        0.3 * cat_prob
    )
    print(f"[OK] Final Ensemble Probability: {final_prob:.4f}")
    return final_prob


def compute_risk_score(probability):
    risk_score = probability * 100
    return round(risk_score, 2)


def classify_risk_level(risk_score):
    if risk_score < 30:
        return "LOW"
    elif risk_score < 60:
        return "MODERATE"
    elif risk_score < 80:
        return "HIGH"
    else:
        return "CRITICAL"