from pathlib import Path

from dotenv import load_dotenv
import os

# Load GEMINI_API_KEY and other secrets from project root .env (not committed).
# override=True so project .env wins over empty/stale GEMINI_* in the OS environment.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.encoders import jsonable_encoder
import joblib
import pandas as pd
import numpy as np
import json
from typing import Any
from io import BytesIO
import re

from tensorflow.keras.models import load_model

from pipeline.feature_engineering import engineer_features
from pipeline.preprocessing import preprocess_data, PreprocessArtifacts
from pipeline.missing_value_engine import (
    DEFAULT_FEATURE_COLUMNS,
    impute_numeric_with_knn,
    flags_to_row_dicts,
    merge_flag_columns,
)
from api.insights import rank_contributing_factors, prevention_recommendations
from models.ensemble_engine import (
    FULL_ENSEMBLE_WEIGHTS_DICT,
    classify_risk_level,
    compute_weighted_ensemble_probability,
)

app = FastAPI()


def _cors_origins_from_env() -> list[str]:
    raw = os.getenv("FRONTEND_ORIGINS", "").strip()
    if not raw:
        return ["*"]
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_from_env(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

MODELS_DIR = Path("models")

BASE_NUMERIC_COLS = list(DEFAULT_FEATURE_COLUMNS)
MAX_EXCEL_ROWS = 100_000


def _ensemble_weighting_meta() -> dict[str, Any]:
    """Explains the weighted linear ensemble returned with batch summaries."""
    return {
        "method": "weighted_linear",
        "weights": dict(FULL_ENSEMBLE_WEIGHTS_DICT),
        "note": (
            "P(ensemble) = Σᵢ wᵢ·pᵢ over eight models with Σwᵢ = 1. "
            "Gradient-boosting trees (xgb, lgbm, cat) carry 40% of total mass in a 40:30:30 split; "
            "the other five models share the remaining 60%."
        ),
    }


def _augment_batch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(summary)
    out["ensemble_weighting"] = _ensemble_weighting_meta()
    rs = out.get("risk_score")
    try:
        rs_f = float(rs)
    except (TypeError, ValueError):
        rs_f = 0.0
    out["risk_level_aggregate"] = classify_risk_level(rs_f)
    return out


PREVIEW_ROW_LIMIT = 2_000
PREDICT_CHUNK = 2_048


def _load_feature_columns() -> list[str]:
    p = MODELS_DIR / "feature_columns.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _load_artifacts() -> PreprocessArtifacts:
    return joblib.load(MODELS_DIR / "preprocess_artifacts.pkl")


def _artifact_mean_fallback() -> dict[str, float]:
    """Column means from training artifacts for single-row / fallback imputation."""
    art = _load_artifacts()
    out: dict[str, float] = {}
    for k, v in art.means.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _drop_empty_and_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(how="all")

    cols = [str(c).strip().lower() for c in df.columns.tolist()]

    def is_header_row(row) -> bool:
        vals = [str(v).strip().lower() for v in row.tolist()]
        return len(vals) == len(cols) and all(vals[i] == cols[i] for i in range(len(cols)))

    mask = df.apply(is_header_row, axis=1)
    if mask.any():
        df = df.loc[~mask]
    return df.reset_index(drop=True)


def _normalize_numeric_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace({"": None, "NULL": None, "null": None, "NaN": None, "nan": None})
    for c in BASE_NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in BASE_NUMERIC_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _slugify_header(name: str) -> str:
    """Turn Excel / human headers into snake_case for matching BASE_NUMERIC_COLS."""
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s)
    s = s.lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _dataframe_rows_for_api(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Build JSON-safe row dicts for cleaned/original previews.
    Avoids json.loads(df.to_json()) which can yield NaN / numpy types that confuse clients.
    """
    if df is None or len(df) == 0:
        return []
    chunk = df.head(PREVIEW_ROW_LIMIT).copy()
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    for c in chunk.columns:
        if str(chunk[c].dtype) == "boolean" or chunk[c].dtype == bool:
            chunk[c] = chunk[c].where(pd.notna(chunk[c]), None)
    chunk = chunk.replace({np.nan: None})
    records = chunk.to_dict(orient="records")
    return jsonable_encoder(records)


def _canonicalize_excel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map spreadsheet columns (e.g. 'Shift Hours', 'ShiftHours') onto shift_hours, etc.
    Produces exactly the eight safety feature columns so charts and KNN see real values.
    """
    by_slug: dict[str, list[str]] = {}
    for c in df.columns:
        slug = _slugify_header(str(c))
        if slug in BASE_NUMERIC_COLS:
            by_slug.setdefault(slug, []).append(c)

    out = pd.DataFrame(index=df.index)
    for slug in BASE_NUMERIC_COLS:
        if slug not in by_slug:
            out[slug] = np.nan
            continue
        cols = by_slug[slug]
        if len(cols) == 1:
            out[slug] = pd.to_numeric(df[cols[0]], errors="coerce")
        else:
            ser = pd.Series(np.nan, index=df.index, dtype=float)
            for oc in cols:
                ser = ser.combine_first(pd.to_numeric(df[oc], errors="coerce"))
            out[slug] = ser
    return out


def _prepare_features(df_in: pd.DataFrame) -> pd.DataFrame:
    artifacts = _load_artifacts()
    feature_columns = _load_feature_columns()

    df = _normalize_numeric_inputs(df_in)
    df, _ = preprocess_data(df, artifacts=artifacts, fit=False)
    df = engineer_features(df)

    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0
    df = df[feature_columns]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def _row_dict_to_dataframe(data: dict) -> pd.DataFrame:
    """Single manual submission: ensure all base columns exist (NaN if omitted)."""
    row: dict[str, Any] = {}
    for c in BASE_NUMERIC_COLS:
        if c not in data or data[c] is None or data[c] == "":
            row[c] = np.nan
        else:
            row[c] = data[c]
    return pd.DataFrame([row])


def _predict_from_features(X: pd.DataFrame) -> pd.DataFrame:
    xgb_prob = xgb.predict_proba(X)[:, 1]
    lgbm_prob = lgbm.predict_proba(X)[:, 1]
    cat_prob = cat.predict_proba(X)[:, 1]
    hgb_prob = hgb.predict_proba(X)[:, 1]
    extra_prob = extra.predict_proba(X)[:, 1]
    stack_prob = stacking.predict_proba(X)[:, 1]

    X_lstm = np.array(X).reshape((X.shape[0], 1, X.shape[1]))
    lstm_prob = lstm.predict(X_lstm, verbose=0).reshape(-1)

    iso_raw = -iso.decision_function(X)
    if len(iso_raw) > 1 and float(np.nanmax(iso_raw) - np.nanmin(iso_raw)) > 1e-12:
        iso_score = (iso_raw - np.nanmin(iso_raw)) / (np.nanmax(iso_raw) - np.nanmin(iso_raw))
    else:
        iso_score = np.clip(iso_raw, 0, 1)

    ensemble_tree = 0.4 * xgb_prob + 0.3 * lgbm_prob + 0.3 * cat_prob

    ensemble_all = compute_weighted_ensemble_probability(
        xgb_prob,
        lgbm_prob,
        cat_prob,
        hgb_prob,
        extra_prob,
        lstm_prob,
        iso_score,
        stack_prob,
    )

    out = pd.DataFrame(
        {
            "xgb_probability": np.round(xgb_prob, 4),
            "lgbm_probability": np.round(lgbm_prob, 4),
            "cat_probability": np.round(cat_prob, 4),
            "hgb_probability": np.round(hgb_prob, 4),
            "extra_trees_probability": np.round(extra_prob, 4),
            "lstm_probability": np.round(lstm_prob, 4),
            "iso_anomaly_score": np.round(iso_score, 4),
            "stacking_probability": np.round(stack_prob, 4),
            "ensemble_tree_probability": np.round(ensemble_tree, 4),
            "ensemble_probability": np.round(ensemble_all, 4),
        }
    )

    out["risk_score"] = (out["ensemble_probability"] * 100).round(2)
    out["risk_level"] = out["risk_score"].apply(classify_risk_level)
    return out


xgb = joblib.load(MODELS_DIR / "xgb.pkl")
lgbm = joblib.load(MODELS_DIR / "lgbm.pkl")
cat = joblib.load(MODELS_DIR / "cat.pkl")
iso = joblib.load(MODELS_DIR / "iso.pkl")
lstm = load_model(MODELS_DIR / "lstm.keras")
hgb = joblib.load(MODELS_DIR / "hgb.pkl")
extra = joblib.load(MODELS_DIR / "extra_trees.pkl")
stacking = joblib.load(MODELS_DIR / "stacking.pkl")


@app.get("/")
def home():
    return {"message": "AccidentZero AI Backend Running"}


@app.api_route("/predict", methods=["POST", "OPTIONS"])
def predict(data: dict):
    # Manual entry: fill missing with training means; KNN not applicable for n=1
    fallback = _artifact_mean_fallback()
    df = _row_dict_to_dataframe(data)
    df = _normalize_numeric_inputs(df)
    df = _ensure_base_columns(df)
    df_before_fill = df.copy()
    imputed_fields: dict[str, bool] = {}
    for c in BASE_NUMERIC_COLS:
        if pd.isna(df[c].iloc[0]):
            df.loc[0, c] = fallback.get(c, 0.0)
            imputed_fields[c] = True
        else:
            imputed_fields[c] = False

    X = _prepare_features(df)
    out = _predict_from_features(X).iloc[0].to_dict()
    out["imputation"] = {
        "fields": imputed_fields,
        "method": "training_mean",
        "note": "Single-row inputs use stored training means for missing cells (KNN requires multiple rows).",
    }
    out["cleaned_inputs"] = {c: float(df[c].iloc[0]) for c in BASE_NUMERIC_COLS if c in df.columns}
    out["original_rows_preview"] = _dataframe_rows_for_api(df_before_fill)
    out["cleaned_rows_preview"] = _dataframe_rows_for_api(df)
    return out


@app.post("/predict/batch")
def predict_batch(rows: list[dict[str, Any]]):
    if not rows:
        return {"rows": [], "summary": {}, "count": 0, "imputation": {"method": "none", "total_imputed_cells": 0}}

    fallback = _artifact_mean_fallback()
    df = pd.DataFrame(rows)
    df = _canonicalize_excel_columns(df)
    df = _normalize_numeric_inputs(df)
    df = _ensure_base_columns(df)

    df_original = df.copy()
    df_clean, flags_df, imp_meta = impute_numeric_with_knn(
        df, feature_cols=BASE_NUMERIC_COLS, n_neighbors=5, fallback_fill=fallback
    )

    parts = []
    for start in range(0, len(df_clean), PREDICT_CHUNK):
        sub = df_clean.iloc[start : start + PREDICT_CHUNK]
        X = _prepare_features(sub)
        parts.append(_predict_from_features(X))
    preds = pd.concat(parts, ignore_index=True)

    summary = _augment_batch_summary(preds.mean(numeric_only=True).round(4).to_dict())
    merged = merge_flag_columns(df_clean, flags_df)

    return {
        "rows": preds.to_dict(orient="records"),
        "summary": summary,
        "count": len(preds),
        "imputation": imp_meta,
        "original_rows_preview": _dataframe_rows_for_api(df_original),
        "cleaned_rows_preview": _dataframe_rows_for_api(merged),
        "flags_per_row": flags_to_row_dicts(flags_df),
    }


@app.get("/correlation")
def get_correlation():
    """Fallback: correlation from bundled sample data if present; else identity."""
    cols = list(BASE_NUMERIC_COLS)
    p = Path("data/safety_data.csv")
    if p.exists():
        try:
            raw = pd.read_csv(p, nrows=5000)
            raw = _normalize_numeric_inputs(raw)
            raw = _ensure_base_columns(raw)
            sub = raw[cols].apply(pd.to_numeric, errors="coerce")
            if sub.shape[1] >= 2 and sub.dropna(how="all").shape[0] >= 2:
                c = sub.corr(numeric_only=True)
                mat = c.reindex(index=cols, columns=cols).fillna(0).values.tolist()
                return {"columns": cols, "matrix": mat, "source": "data/safety_data.csv"}
        except Exception:
            pass
    matrix = np.eye(len(cols)).tolist()
    return {"columns": cols, "matrix": matrix, "source": "identity_fallback"}


@app.post("/correlation")
def correlation_from_uploaded_rows(payload: dict):
    """Correlation on cleaned client / server dataset (preferred for Excel workflow)."""
    rows = payload.get("rows") or []
    if len(rows) < 2:
        raise HTTPException(status_code=400, detail="At least 2 rows are required for correlation.")
    df = pd.DataFrame(rows)
    df = _normalize_numeric_inputs(df)
    df = _ensure_base_columns(df)
    cols = [c for c in BASE_NUMERIC_COLS if c in df.columns]
    if len(cols) < 2:
        raise HTTPException(status_code=400, detail="Need at least two numeric feature columns.")
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    if sub.dropna(how="all").shape[0] < 2:
        raise HTTPException(status_code=400, detail="Not enough valid numeric values.")
    c = sub.corr(numeric_only=True)
    mat = c.reindex(index=cols, columns=cols).fillna(0).values.tolist()
    return {"columns": cols, "matrix": mat, "source": "uploaded_rows"}


@app.post("/predict/excel")
async def predict_excel(file: UploadFile = File(...)):
    try:
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 25 MB).")

        try:
            sheets = pd.read_excel(BytesIO(content), sheet_name=None, engine="openpyxl")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Excel file: {e}") from e

        frames = []
        for _, sdf in sheets.items():
            if sdf is None or sdf.empty:
                continue
            frames.append(_drop_empty_and_repeated_header_rows(sdf))
        if not frames:
            return jsonable_encoder(
                {
                    "rows": [],
                    "summary": {},
                    "count": 0,
                    "imputation": {"method": "empty", "total_imputed_cells": 0},
                }
            )

        df = pd.concat(frames, ignore_index=True)
        if len(df) > MAX_EXCEL_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"Too many rows ({len(df)}). Maximum supported is {MAX_EXCEL_ROWS}.",
            )

        df = _canonicalize_excel_columns(df)
        df = _normalize_numeric_inputs(df)
        df = _ensure_base_columns(df)

        df_original = df.copy()
        fallback = _artifact_mean_fallback()
        df_clean, flags_df, imp_meta = impute_numeric_with_knn(
            df, feature_cols=BASE_NUMERIC_COLS, n_neighbors=5, fallback_fill=fallback
        )

        parts = []
        for start in range(0, len(df_clean), PREDICT_CHUNK):
            sub = df_clean.iloc[start : start + PREDICT_CHUNK]
            X = _prepare_features(sub)
            parts.append(_predict_from_features(X))
        preds = pd.concat(parts, ignore_index=True)

        summary = _augment_batch_summary(preds.mean(numeric_only=True).round(4).to_dict())
        merged = merge_flag_columns(df_clean, flags_df)

        return jsonable_encoder(
            {
                "rows": preds.to_dict(orient="records"),
                "summary": summary,
                "count": len(preds),
                "imputation": imp_meta,
                "original_rows_preview": _dataframe_rows_for_api(df_original),
                "cleaned_rows_preview": _dataframe_rows_for_api(merged),
                "flags_per_row": flags_to_row_dicts(flags_df),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel prediction failed: {e}") from e


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(content) > 12 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 12 MB).")

    try:
        from PIL import Image

        im = Image.open(BytesIO(content))
        im.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    try:
        from api.gemini_image import analyze_image_with_gemini
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="Gemini image module failed to load. Check api/gemini_image.py and dependencies.",
        ) from e

    try:
        result = analyze_image_with_gemini(content)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {e}") from e

    return result


MAX_ANALYSIS_IMAGES = 16


@app.post("/analyze/images")
async def analyze_images(request: Request):
    """
    Analyze up to MAX_ANALYSIS_IMAGES scene photos (Gemini per image, sequential).

    Multipart: repeat form field **files** for each image (browser FormData.append('files', blob, name)).
    Single-file clients may send **file** instead; we accept that too.
    Using Request.form() avoids FastAPI/Starlette edge cases binding list[UploadFile] from some browsers.
    """
    form = await request.form()
    uploads = form.getlist("files")
    if not uploads:
        single = form.get("file")
        if single is not None:
            uploads = [single]
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="No image files provided. Send multipart field 'files' once per image (or 'file' for a single upload).",
        )
    if len(uploads) > MAX_ANALYSIS_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images (max {MAX_ANALYSIS_IMAGES} at once).",
        )

    try:
        from api.gemini_image import analyze_image_with_gemini
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail="Gemini image module failed to load. Check api/gemini_image.py and dependencies.",
        ) from e

    per_image: list[dict[str, Any]] = []
    for upload in uploads:
        if not hasattr(upload, "read"):
            raise HTTPException(
                status_code=400,
                detail="Invalid multipart part (expected file uploads under 'files').",
            )
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=400, detail=f"Empty file: {upload.filename or 'unknown'}.")
        if len(content) > 12 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"Image too large (max 12 MB): {upload.filename}.")

        try:
            from PIL import Image

            im = Image.open(BytesIO(content))
            im.load()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image {upload.filename}: {e}") from e

        try:
            one = analyze_image_with_gemini(content)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Gemini request failed ({upload.filename}): {e}") from e

        one["source_filename"] = upload.filename or ""
        per_image.append(one)

    pcts = [float(p.get("risk_probability_percent", 0)) for p in per_image]
    agg_pct = float(sum(pcts) / len(pcts)) if pcts else 0.0
    agg_pct = max(0.0, min(100.0, round(agg_pct, 2)))
    agg_level = classify_risk_level(agg_pct)

    return {
        "image_count": len(per_image),
        "per_image": per_image,
        "aggregate_risk_probability_percent": agg_pct,
        "aggregate_risk_level": agg_level,
        "aggregate_explanation": (
            f"Arithmetic mean of {len(per_image)} independent scene risk scores (each image analyzed separately)."
        ),
        "accident_risk_factors": _merge_image_factors(per_image),
    }


def _merge_image_factors(per_image: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in per_image:
        for f in p.get("accident_risk_factors") or []:
            s = str(f).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
    return out[:24]


@app.post("/insights/accident")
def accident_insights(body: dict):
    cleaned_rows = body.get("cleaned_rows") or []
    pred_rows = body.get("prediction_rows") or []
    image_analysis = body.get("image_analysis")

    # Image-only path (no tabular batch yet)
    if image_analysis and not cleaned_rows and not pred_rows:
        factors = image_analysis.get("accident_risk_factors") or []
        if not factors and image_analysis.get("per_image"):
            factors = _merge_image_factors(image_analysis["per_image"])
        ranked = [
            {"factor": str(f), "score": max(0.0, 1.0 - i * 0.05), "method": "image"}
            for i, f in enumerate(factors[:12])
        ]
        if not ranked:
            ranked = [{"factor": "scene_hazard", "score": 1.0, "method": "image"}]
        top = ranked[0]["factor"]
        img_for_prev = dict(image_analysis)
        if image_analysis.get("aggregate_risk_probability_percent") is not None:
            img_for_prev["risk_probability_percent"] = image_analysis["aggregate_risk_probability_percent"]
            img_for_prev["risk_level"] = image_analysis.get("aggregate_risk_level") or image_analysis.get(
                "risk_level"
            )
        prevention = prevention_recommendations(ranked, image_summary=img_for_prev, data_risk_level=None)
        return {
            "ranked_factors": ranked,
            "top_cause": top,
            "data_risk_level": None,
            "prevention_recommendations": prevention,
        }

    if not cleaned_rows or not pred_rows:
        raise HTTPException(
            status_code=400,
            detail="Provide cleaned_rows and prediction_rows, or image_analysis alone.",
        )

    n = min(len(cleaned_rows), len(pred_rows))
    pdf = pd.DataFrame(cleaned_rows[:n])
    pdf = _normalize_numeric_inputs(_ensure_base_columns(pdf))

    risk_vals = []
    for r in pred_rows[:n]:
        rs = r.get("risk_score")
        if rs is None:
            rs = (r.get("ensemble_probability") or 0) * 100
        risk_vals.append(float(rs))
    risk = pd.Series(risk_vals)

    ranked = rank_contributing_factors(pdf, risk, feature_cols=BASE_NUMERIC_COLS)
    top = ranked[0]["factor"] if ranked else ""

    avg_risk = float(risk.mean()) if len(risk) else 0.0
    data_lvl = classify_risk_level(avg_risk)

    prevention = prevention_recommendations(
        ranked,
        image_summary=image_analysis,
        data_risk_level=data_lvl,
    )

    return {
        "ranked_factors": ranked,
        "top_cause": top,
        "data_risk_level": data_lvl,
        "prevention_recommendations": prevention,
    }
