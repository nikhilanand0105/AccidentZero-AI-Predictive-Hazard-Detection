"""
Contributing-factor ranking and prevention copy from cleaned data + optional image analysis.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
import pandas as pd

from pipeline.missing_value_engine import DEFAULT_FEATURE_COLUMNS

# Prevention list sizing: many concise bullets; never fewer than this after merge/pad.
PREVENTION_MIN_BULLETS = 6
PREVENTION_MAX_BULLETS = 22

_PREVENTION_SYSTEM = """You are a senior industrial safety advisor for AccidentZero AI.
Output: one JSON object only. No markdown fences, no text outside JSON.
Key "recommendations": array of strings (each string = one bullet line).

Count rules (mandatory):
- Always output AT LEAST 6 distinct bullets, including when dataset_risk_level is LOW or risk is dominated by image-only context.
- If risk is LOW or MODERATE: produce 6–12 bullets (sustain controls, monitoring, housekeeping, training, documentation—still specific, not vague platitudes).
- If risk is HIGH or CRITICAL: produce 14–22 bullets covering escalation, immediate controls, verification, and follow-up.
- Image/scene context: add targeted bullets only when scene data is present.

Style rules:
- Each bullet: start with an imperative verb; max ~35 words; one sentence preferred; no filler ("it is important", "robust", "holistic").
- Vary control types (eliminate/substitute, engineering, administrative, PPE) across bullets; do not duplicate the same control in different wording.
- Industrial / logistics / outdoor work; no medical diagnosis.
"""


def _prevention_count_hint(data_risk_level: str | None) -> str:
    u = (data_risk_level or "").strip().upper()
    if u in ("HIGH", "CRITICAL"):
        return "Aim for 14–22 recommendations (many points, each line actionable)."
    if u in ("LOW",):
        return "Risk is LOW: still output at least 6 and up to 12 recommendations (sustainment, audits, monitoring, training)."
    if u in ("MODERATE",):
        return "Aim for 10–16 recommendations."
    return "Output at least 6 recommendations; if context suggests higher risk, use more (up to 22)."


def _prevention_user_payload(
    ranked_factors: list[dict[str, Any]],
    *,
    image_summary: dict[str, Any] | None,
    data_risk_level: str | None,
) -> str:
    ctx: dict[str, Any] = {
        "ranked_contributing_factors": [
            {"factor": r.get("factor"), "score": r.get("score"), "method": r.get("method")}
            for r in (ranked_factors or [])[:10]
        ],
        "dataset_risk_level": data_risk_level,
        "bullet_count_instruction": _prevention_count_hint(data_risk_level),
    }
    if image_summary:
        ctx["scene_analysis"] = {
            "aggregate_risk_level": image_summary.get("aggregate_risk_level") or image_summary.get("risk_level"),
            "aggregate_risk_percent": image_summary.get("aggregate_risk_probability_percent")
            or image_summary.get("risk_probability_percent"),
            "accident_risk_factors": (image_summary.get("accident_risk_factors") or [])[:12],
            "weather_hazards": (image_summary.get("weather_hazards") or [])[:8],
            "equipment_condition": (image_summary.get("equipment_condition") or "")[:400],
        }
    return (
        "Context (JSON):\n"
        + json.dumps(ctx, indent=2)
        + '\n\nReturn exactly: {"recommendations": ["...", "..."]}'
    )


def _normalize_bullet_line(s: str) -> str:
    t = str(s).strip()
    t = re.sub(r"^[\-\u2022\*]\s*", "", t)
    t = re.sub(r"^\d+[\.\)]\s*", "", t)
    return t.strip()


def _parse_prevention_json(raw: str) -> list[str] | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    rec = data.get("recommendations")
    if rec is None:
        rec = data.get("prevention_recommendations")
    if not isinstance(rec, list):
        return None
    out: list[str] = []
    for item in rec:
        if isinstance(item, str):
            line = _normalize_bullet_line(item)
            if line and line not in out:
                out.append(line[:500])
        elif isinstance(item, dict):
            t = item.get("action") or item.get("text") or item.get("recommendation")
            if isinstance(t, str):
                line = _normalize_bullet_line(t)
                if line and line not in out:
                    out.append(line[:500])
    return out if out else None


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in lines:
        key = s.strip().lower()[:160]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(s.strip())
    return out


def _pad_prevention_list(
    primary: list[str],
    ranked_factors: list[dict[str, Any]],
    *,
    image_summary: dict[str, Any] | None,
    data_risk_level: str | None,
    include_rules_fallback: bool = True,
) -> list[str]:
    """Ensure at least PREVENTION_MIN_BULLETS lines. Optionally merge static rule bullets (for short LLM output)."""
    merged = _dedupe_preserve_order(list(primary))
    if len(merged) >= PREVENTION_MIN_BULLETS:
        return merged[:PREVENTION_MAX_BULLETS]
    fill: list[str] = []
    if include_rules_fallback:
        fill.extend(
            _prevention_recommendations_rules(
                ranked_factors,
                image_summary=image_summary,
                data_risk_level=data_risk_level,
            )
        )
    fill.extend(_maintenance_prevention_baseline())
    merged.extend(fill)
    merged = _dedupe_preserve_order(merged)
    return merged[:PREVENTION_MAX_BULLETS]


def _maintenance_prevention_baseline() -> list[str]:
    """Short, universal lines to pad low-risk or short model output."""
    return [
        "Keep daily/weekly safety inspections on a fixed schedule and log findings with owner and due date.",
        "Review JSA/JHA and permit-to-work for non-routine tasks before each shift that runs them.",
        "Maintain accessible SDS and tool/equipment registers; pull marginal gear from service until repaired.",
        "Track near-misses and first-aid cases in one register; review trends with supervision monthly.",
        "Re-train on stop-work authority and hazard reporting; confirm every employee knows the channel.",
        "Verify emergency stops, guarding, and lockout/tagout points during routine PMs, not only after incidents.",
        "After any process change, re-run this risk analysis and update controls before full production volume.",
    ]


def _prevention_via_gemini(
    ranked_factors: list[dict[str, Any]],
    *,
    image_summary: dict[str, Any] | None,
    data_risk_level: str | None,
) -> list[str] | None:
    flag = (os.environ.get("GEMINI_PREVENTION") or "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return None
    try:
        from api.gemini_image import generate_text_with_gemini
    except ImportError:
        return None
    user = _prevention_user_payload(ranked_factors, image_summary=image_summary, data_risk_level=data_risk_level)
    try:
        try:
            _mt = int((os.environ.get("GEMINI_PREVENTION_MAX_OUTPUT_TOKENS") or "3072").strip())
        except (TypeError, ValueError):
            _mt = 3072
        raw = generate_text_with_gemini(
            system_instruction=_PREVENTION_SYSTEM,
            user_text=user,
            max_output_tokens=max(1024, _mt),
        )
    except Exception:
        return None
    rec = _parse_prevention_json(raw)
    if not rec:
        return None
    return _pad_prevention_list(rec, ranked_factors, image_summary=image_summary, data_risk_level=data_risk_level)


def rank_contributing_factors(
    df: pd.DataFrame,
    risk_scores: pd.Series | np.ndarray,
    *,
    feature_cols: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Rank features by absolute Pearson correlation with risk score (batch).
    Single-row: uses deviation from column means as a proxy.
    """
    feature_cols = [c for c in (feature_cols or DEFAULT_FEATURE_COLUMNS) if c in df.columns]
    if not feature_cols:
        return []

    rs = np.asarray(risk_scores, dtype=float).reshape(-1)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    if len(rs) < 2 or np.nanstd(rs) < 1e-12:
        # Degenerate: rank by absolute z-scores vs column means
        means = X.mean(numeric_only=True)
        stds = X.std(numeric_only=True).replace(0, np.nan)
        row0 = X.iloc[0]
        scores = []
        for c in feature_cols:
            z = abs((row0[c] - means[c]) / stds[c]) if pd.notna(stds[c]) else abs(row0[c] - means[c])
            scores.append({"factor": c, "score": float(z) if pd.notna(z) else 0.0, "method": "z_deviation"})
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores

    corrs = []
    for c in feature_cols:
        col = pd.to_numeric(X[c], errors="coerce")
        pair = pd.DataFrame({"x": col, "r": rs}).dropna()
        if len(pair) < 2:
            corrs.append({"factor": c, "score": 0.0, "method": "pearson"})
            continue
        v = pair["x"].corr(pair["r"])
        corrs.append({"factor": c, "score": float(abs(v)) if pd.notna(v) else 0.0, "method": "pearson"})

    corrs.sort(key=lambda x: x["score"], reverse=True)
    return corrs


def _prevention_recommendations_rules(
    ranked_factors: list[dict[str, Any]],
    *,
    image_summary: dict[str, Any] | None = None,
    data_risk_level: str | None = None,
) -> list[str]:
    """Static fallback when Gemini is disabled or unavailable."""
    bullets: list[str] = []
    top = ranked_factors[0]["factor"] if ranked_factors else None

    if top == "overtime_hours" or top == "shift_hours":
        bullets.append(
            "Review scheduling: reduce consecutive long shifts and cap overtime where fatigue metrics are elevated."
        )
    if top == "equipment_age" or top == "maintenance_score" or top == "inspection_score":
        bullets.append(
            "Prioritize equipment inspections and preventive maintenance on the highest-risk assets identified in the data."
        )
    if top == "temperature" or top == "humidity":
        bullets.append(
            "Adjust environmental controls or work/rest cycles during extreme temperature or humidity conditions."
        )
    if top == "worker_experience":
        bullets.append(
            "Pair less experienced operators with mentors and reinforce standard work during higher-risk periods."
        )

    if image_summary:
        lvl = (image_summary.get("risk_level") or "").lower()
        if "high" in lvl or "critical" in lvl:
            bullets.append(
                "Address visible hazards from the uploaded scene before resuming work; verify guarding and housekeeping."
            )
        hazards = image_summary.get("weather_hazards") or []
        if hazards:
            bullets.append(
                f"Weather / environment: monitor {', '.join(hazards[:3])} and postpone exposed tasks if conditions worsen."
            )
        equip = image_summary.get("equipment_condition") or ""
        if isinstance(equip, str) and len(equip) > 10:
            bullets.append(
                "Follow up with a focused mechanical inspection based on visible equipment condition cues in the image."
            )

    if data_risk_level and data_risk_level.upper() in ("HIGH", "CRITICAL"):
        bullets.append(
            "Escalate to a formal safety review and document controls before continuing similar operations."
        )

    seen = set()
    out: list[str] = []
    for b in bullets:
        if b not in seen:
            seen.add(b)
            out.append(b)

    if not out:
        out.append(
            "Continue routine monitoring, keep logs updated, and re-run analysis after any process or environment change."
        )

    return out


def prevention_recommendations(
    ranked_factors: list[dict[str, Any]],
    *,
    image_summary: dict[str, Any] | None = None,
    data_risk_level: str | None = None,
) -> list[str]:
    """
    Point-wise prevention and controls: Gemini when enabled (GEMINI_API_KEY + GEMINI_PREVENTION≠0),
    else rule-based fallback. Always at least PREVENTION_MIN_BULLETS lines after merge/pad.
    """
    ai = _prevention_via_gemini(
        ranked_factors,
        image_summary=image_summary,
        data_risk_level=data_risk_level,
    )
    if ai:
        return ai
    raw = _prevention_recommendations_rules(
        ranked_factors,
        image_summary=image_summary,
        data_risk_level=data_risk_level,
    )
    return _pad_prevention_list(
        raw,
        ranked_factors,
        image_summary=image_summary,
        data_risk_level=data_risk_level,
        include_rules_fallback=False,
    )
