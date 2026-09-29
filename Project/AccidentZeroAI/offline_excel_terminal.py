from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api import app as api_app
from api.insights import rank_contributing_factors
from models.ensemble_engine import (
    FULL_ENSEMBLE_WEIGHTS,
    FULL_ENSEMBLE_WEIGHTS_DICT,
    classify_risk_level,
)
from pipeline.missing_value_engine import impute_numeric_with_knn

# Order matches FULL_ENSEMBLE_WEIGHTS / backend _predict_from_features
_MODEL_PROB_COLS = [
    "xgb_probability",
    "lgbm_probability",
    "cat_probability",
    "hgb_probability",
    "extra_trees_probability",
    "lstm_probability",
    "iso_anomaly_score",
    "stacking_probability",
]

_MODEL_LABELS_SHORT = ["XGB", "LGBM", "CAT", "HistGBM", "ExtraTrees", "LSTM", "ISO", "Stacking"]


def _print_section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def _read_excel_as_backend(excel_path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
    frames: list[pd.DataFrame] = []
    for _, sdf in sheets.items():
        if sdf is None or sdf.empty:
            continue
        frames.append(api_app._drop_empty_and_repeated_header_rows(sdf))
    if not frames:
        raise ValueError("Excel contains no usable rows.")
    df = pd.concat(frames, ignore_index=True)
    df = api_app._canonicalize_excel_columns(df)
    df = api_app._normalize_numeric_inputs(df)
    df = api_app._ensure_base_columns(df)
    return df


def _correlation_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Same logic as FastAPI /correlation on uploaded numeric rows."""
    use = [c for c in cols if c in df.columns]
    if len(use) < 2:
        return pd.DataFrame()
    sub = df[use].apply(pd.to_numeric, errors="coerce")
    if sub.dropna(how="all").shape[0] < 2:
        return pd.DataFrame()
    return sub.corr(numeric_only=True)


def _risk_buckets(risk_scores: pd.Series) -> list[int]:
    """Frontend buckets: 0–30, 30–60, 60–80, 80–100."""
    buckets = [0, 0, 0, 0]
    for s in risk_scores.astype(float):
        if s < 30:
            buckets[0] += 1
        elif s < 60:
            buckets[1] += 1
        elif s < 80:
            buckets[2] += 1
        else:
            buckets[3] += 1
    return buckets


def _verify_row0_weighted_sum(preds: pd.DataFrame) -> tuple[float, float, str]:
    """Row 0: manual dot product vs stored ensemble_probability."""
    if len(preds) == 0:
        return 0.0, 0.0, "n/a"
    row = preds.iloc[0]
    vec = np.array([float(row[c]) for c in _MODEL_PROB_COLS], dtype=float)
    manual = float(np.dot(FULL_ENSEMBLE_WEIGHTS, vec))
    stored = float(row["ensemble_probability"])
    ok = abs(manual - stored) < 1e-4
    return manual, stored, "match" if ok else f"diff={abs(manual - stored):.6f}"


def _feature_cols_present(df: pd.DataFrame) -> list[str]:
    return [c for c in api_app.BASE_NUMERIC_COLS if c in df.columns]


def _plot_correlation_heatmap_on_ax(
    ax,
    cm: pd.DataFrame,
    *,
    sns: Any,
    fig: Any,
) -> None:
    """Draw Pearson correlation heatmap on a single axes."""
    if cm.shape[0] < 2:
        ax.text(
            0.5,
            0.5,
            "Need at least 2 numeric columns and 2 data rows\nfor a correlation matrix.",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            cbar_kws={"label": "Pearson r"},
        )
    else:
        im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cm.index)))
        ax.set_yticklabels(cm.index, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")


def _page_numeric_feature_graph(
    plt: Any,
    df: pd.DataFrame,
    *,
    page_i: int,
    total_pages: int,
    headline: str,
    subtitle: str,
    excel_path: Path,
) -> None:
    """One window: line plots of each base numeric column vs row index."""
    cols = _feature_cols_present(df)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Page {page_i}/{total_pages} — {headline}\n{subtitle}\nFile: {excel_path.name}",
        fontsize=12,
        fontweight="bold",
    )
    n_feat = max(len(cols), 1)
    nrows = int(np.ceil(n_feat / 4))
    ncols = min(4, n_feat)
    gs = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.35)

    if not cols:
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0.5, 0.5, "No numeric base columns found in this frame.", ha="center", va="center")
        ax.set_axis_off()
    else:
        x = np.arange(len(df), dtype=float) + 1.0
        for idx, c in enumerate(cols):
            r, col = divmod(idx, 4)
            ax = fig.add_subplot(gs[r, col])
            y = pd.to_numeric(df[c], errors="coerce")
            ax.plot(x, y, marker="o", markersize=4, linewidth=1.2, color="C0")
            ax.set_title(c.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Row index (Excel order)")
            ax.set_ylabel("Feature value")
            ax.grid(True, alpha=0.35)
            if len(df) > 0:
                ax.set_xlim(0.5, len(df) + 0.5)
    plt.tight_layout(rect=[0, 0.02, 1, 0.90], pad=1.2, h_pad=1.8, w_pad=1.2)
    plt.show()
    plt.close(fig)


def _page_correlation_only(
    plt: Any,
    cm: pd.DataFrame,
    *,
    page_i: int,
    total_pages: int,
    title: str,
    footnote: str,
    excel_path: Path,
    sns: Any | None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle(
        f"Page {page_i}/{total_pages} — {title}\nFile: {excel_path.name}",
        fontsize=12,
        fontweight="bold",
    )
    _plot_correlation_heatmap_on_ax(ax, cm, sns=sns, fig=fig)
    ax.set_xlabel("Feature (column)")
    ax.set_ylabel("Feature (row)")
    fig.text(0.5, 0.02, footnote, ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.06, 1, 0.91], pad=1.2, h_pad=1.5)
    plt.show()
    plt.close(fig)


def _show_six_sequential_popups(
    *,
    df_pre: pd.DataFrame,
    df_processed: pd.DataFrame,
    preds: pd.DataFrame,
    summary: dict[str, Any],
    ranked_factors: list[dict[str, Any]],
    excel_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ImportError:
        sns = None

    total_pages = 6
    cols = list(api_app.BASE_NUMERIC_COLS)
    cm_pre = _correlation_matrix(df_pre, cols)
    cm_post = _correlation_matrix(df_processed, cols)

    # --- Page 1: preprocessed — numeric feature graphs ---
    _page_numeric_feature_graph(
        plt,
        df_pre,
        page_i=1,
        total_pages=total_pages,
        headline="Preprocessed data (graphs)",
        subtitle="Values read from Excel after column mapping; before KNN / median imputation.",
        excel_path=excel_path,
    )

    # --- Page 2: preprocessed correlation matrix ---
    _page_correlation_only(
        plt,
        cm_pre,
        page_i=2,
        total_pages=total_pages,
        title="Correlation matrix — preprocessed data",
        footnote="Pearson correlation between base numeric columns (same rules as API /correlation).",
        excel_path=excel_path,
        sns=sns,
    )

    # --- Page 3: processed — numeric feature graphs ---
    _page_numeric_feature_graph(
        plt,
        df_processed,
        page_i=3,
        total_pages=total_pages,
        headline="Processed data (graphs)",
        subtitle="After imputation (KNN when possible; else training-mean fallback). Ready for model input.",
        excel_path=excel_path,
    )

    # --- Page 4: processed correlation matrix ---
    _page_correlation_only(
        plt,
        cm_post,
        page_i=4,
        total_pages=total_pages,
        title="Correlation matrix — processed (imputed) data",
        footnote="Correlations on filled values; use with care when many cells were imputed.",
        excel_path=excel_path,
        sns=sns,
    )

    # --- Page 5: model outputs + attribute (Excel feature) weightage ---
    # Tall figure + large row spacing so titles/xlabels do not overlap (esp. bar vs polar vs bottom panel).
    fig5 = plt.figure(figsize=(15, 15))
    fig5.suptitle(
        f"Page 5/{total_pages} — Model outputs & Excel attribute association with risk\n"
        f"File: {excel_path.name}",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    gs5 = fig5.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.1, 1.35],
        hspace=0.78,
        wspace=0.38,
    )

    ax_bar = fig5.add_subplot(gs5[0, :])
    avgx = [float(summary.get(k, 0) or 0) for k in _MODEL_PROB_COLS]
    avgx.append(float(summary.get("ensemble_probability", 0) or 0))
    labels_bar = _MODEL_LABELS_SHORT + ["Ensemble"]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(labels_bar)))
    bars = ax_bar.bar(labels_bar, avgx, color=colors, edgecolor="black", linewidth=0.35)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Average probability / score (0–1)")
    # Omit xlabel here — model names on ticks are enough — avoids collision with row 2 titles.
    ax_bar.set_title(
        "Batch average — per-model outputs and final ensemble (matches frontend overview chart)",
        pad=10,
    )
    ax_bar.tick_params(axis="x", rotation=20, labelsize=9, pad=6)
    ax_bar.set_xlabel("")  # explicit empty: category labels are self-explanatory
    for b, v in zip(bars, avgx):
        ax_bar.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_risk = fig5.add_subplot(gs5[1, 0])
    buckets = _risk_buckets(preds["risk_score"])
    band_labels = ["0–30\n(LOW)", "30–60\n(MOD)", "60–80\n(HIGH)", "80–100\n(CRIT)"]
    ax_risk.bar(band_labels, buckets, color=plt.cm.Blues(0.55), edgecolor="black", linewidth=0.35)
    ax_risk.set_ylabel("Number of rows in batch")
    # Short axis label; full definition in figure footnote below.
    ax_risk.set_xlabel("Risk band")
    ax_risk.set_title("Risk score distribution (all rows)", pad=8)
    ymax = max(buckets + [1])
    for i, v in enumerate(buckets):
        ax_risk.text(i, v + 0.02 * ymax, str(int(v)), ha="center", fontsize=10)

    ax_polar = fig5.add_subplot(gs5[1, 1], projection="polar")
    if len(preds) > 0:
        row0 = preds.iloc[0]
        vals = [float(row0[c]) for c in _MODEL_PROB_COLS]
        vals.append(float(row0["ensemble_probability"]))
        n = len(_MODEL_LABELS_SHORT + ["Ens"])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        vals_closed = np.concatenate([vals, [vals[0]]])
        angles_closed = np.concatenate([angles, [angles[0]]])
        ax_polar.plot(angles_closed, vals_closed, "o-", linewidth=2)
        ax_polar.fill(angles_closed, vals_closed, alpha=0.22)
        ax_polar.set_xticks(np.linspace(0, 2 * np.pi, n, endpoint=False))
        ax_polar.set_xticklabels(_MODEL_LABELS_SHORT + ["Ens"], fontsize=8)
        ax_polar.set_ylim(0, 1)
        ax_polar.set_title("First Excel row — model scores (0–1)", pad=28)

    ax_attr = fig5.add_subplot(gs5[2, :])
    if ranked_factors:
        names = [str(r.get("factor", "")) for r in ranked_factors]
        scores = [float(r.get("score", 0) or 0) for r in ranked_factors]
        methods = [str(r.get("method", "")) for r in ranked_factors]
        y_pos = np.arange(len(names))
        ax_attr.barh(y_pos, scores, color=plt.cm.viridis(np.linspace(0.2, 0.85, len(names))))
        ax_attr.set_yticks(y_pos)
        ax_attr.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9)
        ax_attr.invert_yaxis()
        ax_attr.set_xlabel("|r| vs batch risk_score (or z-deviation if n<2)")
        ax_attr.set_title("Excel attribute ranking (insights engine)", pad=12)
        xmax = max(scores) if scores else 1.0
        pad_x = max(0.02 * xmax, 0.008)
        for i, (s, m) in enumerate(zip(scores, methods)):
            ax_attr.text(s + pad_x, i, f"{s:.3f} ({m})", va="center", fontsize=8)
    else:
        ax_attr.text(0.5, 0.5, "No ranked factors returned.", ha="center", va="center", transform=ax_attr.transAxes)
    ax_attr.grid(True, axis="x", alpha=0.3)

    fig5.text(
        0.5,
        0.02,
        "risk_score = ensemble_probability × 100  |  Risk bands: 0–30 LOW, 30–60 MOD, 60–80 HIGH, 80–100 CRIT",
        ha="center",
        fontsize=8,
        style="italic",
    )
    # Reserve space for suptitle + bottom footnote without collapsing row gaps from GridSpec.
    fig5.subplots_adjust(top=0.94, bottom=0.06)
    plt.show()
    plt.close(fig5)

    # --- Page 6: final batch output + fixed ensemble weights for each model ---
    manual, stored, chk = _verify_row0_weighted_sum(preds)
    final_prob = float(summary.get("ensemble_probability", 0))
    final_risk = float(summary.get("risk_score", 0))
    final_level = str(summary.get("risk_level_aggregate") or classify_risk_level(final_risk))

    fig6 = plt.figure(figsize=(14, 12))
    fig6.suptitle(
        f"Page 6/{total_pages} — Final aggregate output & ensemble model weights\nFile: {excel_path.name}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    g6 = fig6.add_gridspec(2, 2, height_ratios=[1.15, 1.05], hspace=0.52, wspace=0.42)

    ax_out = fig6.add_subplot(g6[0, 0])
    ax_out.axis("off")
    summary_lines = [
        "BATCH AGGREGATE (mean over rows) — same numbers as frontend summary cards",
        "",
        f"  ensemble_probability  P  = {final_prob:.6f}",
        f"  risk_score            = P × 100 = {final_risk:.6f}",
        f"  risk_level (thresholds) = {final_level}",
        f"  rows in batch = {len(preds)}",
        "",
        "Risk thresholds:  <30 LOW  |  <60 MODERATE  |  <80 HIGH  |  else CRITICAL",
        "",
        f"Row 0 check:  sum_i w_i * p_i = {manual:.6f}",
        f"             stored ensemble = {stored:.6f}  ({chk})",
        "",
        "Tree-only reference (column ensemble_tree_probability):",
        "  0.40*p_xgb + 0.30*p_lgbm + 0.30*p_cat",
    ]
    ax_out.text(
        0.03,
        0.98,
        "\n".join(summary_lines),
        transform=ax_out.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "#e8f4f8", "alpha": 0.95},
    )

    ax_w = fig6.add_subplot(g6[0, 1])
    w_labels = [k.replace("_", "\n") for k in FULL_ENSEMBLE_WEIGHTS_DICT.keys()]
    w_vals = [FULL_ENSEMBLE_WEIGHTS_DICT[k] for k in FULL_ENSEMBLE_WEIGHTS_DICT.keys()]
    x_pos = np.arange(len(w_vals))
    ax_w.bar(x_pos, w_vals, color=plt.cm.Set2(np.linspace(0, 1, len(w_vals))), edgecolor="black", linewidth=0.35)
    ax_w.set_xticks(x_pos)
    ax_w.set_xticklabels(w_labels, fontsize=8)
    ax_w.set_ylabel("Weight w_i")
    ax_w.set_xlabel("Model (fixed in ensemble_engine)")
    ax_w.set_title("Ensemble weights: P = sum_i w_i * p_i (sum = 1)", pad=8)
    for i, v in enumerate(w_vals):
        ax_w.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)

    ax_form = fig6.add_subplot(g6[1, :])
    ax_form.axis("off")
    formula_lines = [
        "ENSEMBLE (8 learned models, 8 fixed weights)",
        "",
        "  P_ensemble = w_xgb*p_xgb + w_lgbm*p_lgbm + w_cat*p_cat + w_hgb*p_hgb",
        "             + w_extra*p_extra + w_lstm*p_lstm + w_iso*s_iso + w_stack*p_stack",
        "",
        "  where each p_* is in [0,1] (ISO score normalized to [0,1] in the pipeline).",
        "",
        "Per-model weights (FULL_ENSEMBLE_WEIGHTS_DICT):",
    ]
    for k, w in FULL_ENSEMBLE_WEIGHTS_DICT.items():
        formula_lines.append(f"    {k:16s}  w = {w:.6f}")
    formula_lines.extend(
        [
            "",
            f"Verification: sum of displayed weights = {sum(FULL_ENSEMBLE_WEIGHTS_DICT.values()):.6f}",
        ]
    )
    ax_form.text(
        0.02,
        0.98,
        "\n".join(formula_lines),
        transform=ax_form.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.92},
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.89], pad=2.0, h_pad=2.8, w_pad=2.5)
    plt.show()
    plt.close(fig6)


def run_offline_excel(excel_path: Path, preview_rows: int = 10, *, show_plots: bool = True) -> None:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    _print_section("OFFLINE EXCEL PIPELINE (TERMINAL MODE)")
    print(f"Input file: {excel_path}")
    print("Image analysis is excluded in this mode.")

    # 1) Preprocessed data (canonicalized numeric inputs before imputation)
    df_pre = _read_excel_as_backend(excel_path)
    _print_section("1) PREPROCESSED DATA (BEFORE IMPUTATION)")
    print(f"Rows: {len(df_pre)} | Columns: {len(df_pre.columns)}")
    print(df_pre.head(preview_rows).to_string(index=False))

    # 2) Processed data (after KNN/median fallback imputation)
    fallback = api_app._artifact_mean_fallback()
    df_processed, flags_df, imp_meta = impute_numeric_with_knn(
        df_pre,
        feature_cols=api_app.BASE_NUMERIC_COLS,
        n_neighbors=5,
        fallback_fill=fallback,
    )
    _print_section("2) PROCESSED DATA (AFTER IMPUTATION / CLEANING)")
    print(f"Rows: {len(df_processed)} | Columns: {len(df_processed.columns)}")
    print(f"Imputation meta: {imp_meta}")
    print(df_processed.head(preview_rows).to_string(index=False))

    # Features exactly as backend predicts
    X = api_app._prepare_features(df_processed)
    _print_section("3) MODEL INPUT FEATURES (AFTER PREPROCESS + FEATURE ENGINEERING)")
    print(f"Rows: {X.shape[0]} | Features: {X.shape[1]}")
    print("Feature columns:", list(X.columns))
    print(X.head(preview_rows).to_string(index=False))

    # 3) Individual model outputs + final output from backend inference function
    preds = api_app._predict_from_features(X)
    _print_section("4) OUTPUT OF ALL MODELS INDIVIDUALLY")
    model_cols = [
        "xgb_probability",
        "lgbm_probability",
        "cat_probability",
        "hgb_probability",
        "extra_trees_probability",
        "lstm_probability",
        "iso_anomaly_score",
        "stacking_probability",
        "ensemble_tree_probability",
        "ensemble_probability",
        "risk_score",
        "risk_level",
    ]
    print(preds[model_cols].head(preview_rows).to_string(index=False))

    summary = preds.mean(numeric_only=True).round(4).to_dict()
    summary = api_app._augment_batch_summary(summary)
    final_prob = float(summary.get("ensemble_probability", 0.0))
    final_risk = float(summary.get("risk_score", 0.0))
    final_level = str(summary.get("risk_level_aggregate") or classify_risk_level(final_risk))

    ranked = rank_contributing_factors(
        df_processed,
        preds["risk_score"],
        feature_cols=api_app.BASE_NUMERIC_COLS,
    )

    _print_section("5) FINAL AGGREGATE OUTPUT (MATCHES FRONTEND NUMBERS)")
    print(f"Accident Probability (avg ensemble_probability): {final_prob:.4f}")
    print(f"Risk Score (avg risk_score): {final_risk:.4f}")
    print(f"Risk Level (from thresholds): {final_level}")
    print(f"Batch rows: {len(preds)}")

    _print_section("6) FORMULAS / WEIGHTAGE USED")
    print("Tree-only ensemble (for reference):")
    print("  ensemble_tree_probability = 0.4*xgb + 0.3*lgbm + 0.3*cat")
    print()
    print("Final ensemble (used for frontend/main output):")
    print("  ensemble_probability = sum(w_i * p_i), where sum(w_i) = 1")
    print(f"  weights = {FULL_ENSEMBLE_WEIGHTS_DICT}")
    print()
    print("Risk conversion:")
    print("  risk_score = ensemble_probability * 100")
    print("  risk_level: <30 LOW, <60 MODERATE, <80 HIGH, else CRITICAL")

    _print_section("7) FRONTEND CONSISTENCY CHECK")
    print("Frontend non-graph cards are sourced from these same fields:")
    print("  Probability <- summary.ensemble_probability")
    print("  Risk Score <- summary.risk_score")
    print("  Risk Level <- summary.risk_level_aggregate (same threshold function)")
    print("Result: terminal pipeline and frontend API path use the same formulas.")

    if show_plots:
        print("\n[INFO] Opening 6 matplotlib windows in order. Close each window to open the next.")
        try:
            _show_six_sequential_popups(
                df_pre=df_pre,
                df_processed=df_processed,
                preds=preds,
                summary=summary,
                ranked_factors=ranked,
                excel_path=excel_path,
            )
        except Exception as e:
            print(f"[WARN] Could not show plots: {e}. Install matplotlib/seaborn or use --no-plots.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline terminal processing for Excel input using AccidentZero backend formulas."
    )
    parser.add_argument("excel_path", type=str, help="Path to .xlsx file")
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="How many rows to print per section (default: 10)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not open matplotlib pop-up (terminal text only).",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel_path).expanduser().resolve()
    run_offline_excel(excel_path, preview_rows=max(1, args.preview_rows), show_plots=not args.no_plots)


if __name__ == "__main__":
    main()
