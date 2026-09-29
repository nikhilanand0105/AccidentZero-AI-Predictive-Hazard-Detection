"""
Gemini Vision via Google AI Generative Language REST API (same integration style as Next.js + Genkit):
  POST {GEMINI_API_BASE}/models/{model}:generateContent

Env:
  GEMINI_API_KEY or GOOGLE_API_KEY — required
  GEMINI_API_BASE — optional; default v1beta (see _generative_api_base)
  PARSE_FIELDS_API_BASE / GOOGLE_AI_API_BASE — aliases for GEMINI_API_BASE
  GEMINI_MODEL or GEMINI_VISION_MODEL — optional; default gemini-2.5-flash (strip leading models/)
  GEMINI_MODEL_FALLBACKS — comma-separated extras if primary fails (wrong name, region, quota)

We try each (model × API base v1 then v1beta) with systemInstruction, then merged user prompt.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import requests

_genai_loaded = False


def _ensure_env_loaded() -> None:
    """Load .env from project root so the key is available even if uvicorn cwd differs."""
    global _genai_loaded
    if _genai_loaded:
        return
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parent.parent
        load_dotenv(root / ".env", override=True)
        load_dotenv(override=True)  # cwd .env if present
    except ImportError:
        pass
    _genai_loaded = True


def _generative_api_base() -> str:
    """Default v1beta: newer Gemini models are consistently available on v1beta generateContent."""
    return (
        os.environ.get("GEMINI_API_BASE")
        or os.environ.get("PARSE_FIELDS_API_BASE")
        or os.environ.get("GOOGLE_AI_API_BASE")
        or "https://generativelanguage.googleapis.com/v1beta"
    ).rstrip("/")


def _generative_model_id() -> str:
    """GEMINI_MODEL / GEMINI_VISION_MODEL, or default gemini-2.5-flash (current Flash + vision)."""
    raw = (
        os.environ.get("GEMINI_MODEL")
        or os.environ.get("GEMINI_VISION_MODEL")
        or "gemini-2.5-flash"
    ).strip()
    if raw.startswith("models/"):
        raw = raw.split("/", 1)[1]
    return raw


def _model_ids_to_try() -> list[str]:
    """Primary model plus GEMINI_MODEL_FALLBACKS (comma-separated)."""
    primary = _generative_model_id()
    out: list[str] = [primary]
    extra = os.environ.get(
        "GEMINI_MODEL_FALLBACKS",
        "gemini-2.0-flash",
    )
    for part in extra.split(","):
        m = part.strip().replace("models/", "")
        if m and m not in out:
            out.append(m)
    return out


def _api_bases_to_try() -> list[str]:
    primary = _generative_api_base()
    fallback = (
        os.environ.get("GEMINI_API_FALLBACK_BASE")
        or "https://generativelanguage.googleapis.com/v1beta"
    ).rstrip("/")
    out: list[str] = []
    for b in (primary, fallback):
        if b and b not in out:
            out.append(b)
    return out


def _flatten_system_into_user_message(body: dict[str, Any]) -> dict[str, Any]:
    """If the API rejects systemInstruction, merge system text into the user turn (multimodal still works)."""
    sys_parts = body.get("systemInstruction", {}).get("parts") or []
    sys_text = sys_parts[0].get("text", "") if sys_parts else ""
    user_parts = body["contents"][0]["parts"]
    merged_text = (
        sys_text
        + "\n\n---\nUse the image below. Reply with the JSON object only, no markdown.\n---\n"
    )
    return {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": merged_text}, *user_parts],
            }
        ],
        "generationConfig": body.get("generationConfig", {}),
    }


def _flatten_system_user_text_only(body: dict[str, Any]) -> dict[str, Any]:
    """Text-only: merge systemInstruction + user text into a single user message for API fallbacks."""
    sys_parts = body.get("systemInstruction", {}).get("parts") or []
    sys_text = (sys_parts[0].get("text", "") if sys_parts else "").strip()
    user_parts = body["contents"][0]["parts"]
    texts: list[str] = []
    for p in user_parts:
        if "text" in p:
            texts.append(str(p["text"]))
    user_blob = "\n".join(texts).strip()
    merged = (sys_text + "\n\n---\n\n" + user_blob) if sys_text else user_blob
    return {
        "contents": [{"role": "user", "parts": [{"text": merged.strip()}]}],
        "generationConfig": body.get("generationConfig", {}),
    }


def _run_gemini_generate(
    api_key: str,
    body_with_system: dict[str, Any],
    body_flat: dict[str, Any],
    *,
    timeout: int = 120,
) -> dict[str, Any]:
    """Run generateContent with the same retry matrix as vision (system + merged bodies × models × API bases)."""

    def _post(url: str, payload: dict[str, Any]) -> requests.Response:
        return requests.post(
            url,
            json=payload,
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
        )

    def _post_query_key(url: str, payload: dict[str, Any]) -> requests.Response:
        return requests.post(
            url,
            params={"key": api_key},
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )

    last_err: str | None = None
    payload: dict[str, Any] = {}
    found = False

    for body_label, body in (("systemInstruction", body_with_system), ("merged_user_prompt", body_flat)):
        for model_id in _model_ids_to_try():
            for base in _api_bases_to_try():
                url = _generate_content_url(base, model_id)
                try:
                    r = _post(url, body)
                    if r.status_code in (400, 401, 403):
                        msg = _http_error_message(r).lower()
                        if "api key" in msg:
                            r = _post_query_key(url, body)
                except requests.RequestException as e:
                    last_err = f"{body_label} {model_id} {base}: network {e}"
                    continue
                if r.status_code >= 400:
                    err_text = _http_error_message(r)
                    if _is_invalid_api_key_response(r):
                        raise RuntimeError(
                            "Gemini rejected this API key (invalid, expired, or restricted). "
                            "Create a new key at https://aistudio.google.com/apikey , put it in the project .env as GEMINI_API_KEY, "
                            "and restart the API. Ensure the Generative Language API is enabled for that key's Google Cloud project. "
                            f"Google said: {err_text[:500]}"
                        )
                    if _is_quota_exceeded_response(r):
                        raise RuntimeError(
                            "GEMINI_QUOTA_EXCEEDED: Gemini quota/rate-limit reached for this API key. "
                            "Enable billing/increase quota or retry after reset."
                        )
                    last_err = f"{body_label} {model_id} {base}: {err_text}"
                    continue
                try:
                    payload = r.json()
                except json.JSONDecodeError:
                    last_err = "Invalid JSON from Gemini"
                    continue
                if not payload.get("candidates"):
                    fb = payload.get("promptFeedback") or {}
                    br = fb.get("blockReason", "")
                    last_err = f"No candidates (blockReason={br or 'empty'})"
                    continue
                found = True
                break
            if found:
                break
        if found:
            break

    if not found:
        raise RuntimeError(
            last_err
            or "Gemini request failed after retries. Check API key, enable Generative Language API, and billing."
        )
    return payload


def _gemini_payload_text(payload: dict[str, Any]) -> str:
    text = ""
    for cand in payload.get("candidates", []):
        for part in cand.get("content", {}).get("parts") or []:
            if "text" in part:
                text += str(part["text"])
    return text.strip()


def generate_text_with_gemini(
    *,
    system_instruction: str,
    user_text: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> str:
    """
    Plain-text Gemini call (no image). Used for prevention / recommendation bullets and similar.
    """
    _ensure_env_loaded()
    api_key = _normalize_api_key(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")

    t = temperature
    if t is None:
        t = _env_float("GEMINI_PREVENTION_TEMPERATURE", _env_float("GEMINI_TEMPERATURE", 0.35))
    mtok = max_output_tokens if max_output_tokens is not None else _env_int("GEMINI_PREVENTION_MAX_OUTPUT_TOKENS", 1536)

    body_with_system: dict[str, Any] = {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system_instruction.strip()}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text.strip()}],
            }
        ],
        "generationConfig": {
            "temperature": t,
            "maxOutputTokens": mtok,
        },
    }
    body_flat = _flatten_system_user_text_only(body_with_system)
    try:
        payload = _run_gemini_generate(api_key, body_with_system, body_flat)
    except RuntimeError as e:
        if "GEMINI_QUOTA_EXCEEDED" in str(e):
            return _local_quota_fallback(image_bytes)
        raise
    return _gemini_payload_text(payload)


def _http_error_message(resp: requests.Response) -> str:
    try:
        j = resp.json()
        err = j.get("error", {})
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(j.get("detail"), str):
            return j["detail"]
    except Exception:
        pass
    return (resp.text or "")[:800]


def _is_invalid_api_key_response(resp: requests.Response) -> bool:
    """Google returns 400/403 with a message when the key is wrong, expired, or restricted."""
    if resp.status_code not in (400, 401, 403):
        return False
    msg = _http_error_message(resp).lower()
    if "api key" not in msg and "apikey" not in msg:
        return False
    return any(
        x in msg
        for x in (
            "invalid",
            "not valid",
            "expired",
            "api_key_invalid",
            "did not provide",
            "must be used with",
        )
    )


def _is_quota_exceeded_response(resp: requests.Response) -> bool:
    """Detect quota/rate-limit/billing responses from Gemini."""
    if resp.status_code not in (400, 403, 429):
        return False
    msg = _http_error_message(resp).lower()
    return any(
        k in msg
        for k in (
            "quota exceeded",
            "rate limit",
            "free_tier_requests",
            "generate_content_free_tier_requests",
            "billing",
            "retry in",
        )
    )


def _generate_content_url(api_base: str, model_id: str) -> str:
    return f"{api_base.rstrip('/')}/models/{model_id}:generateContent"


def _normalize_api_key(raw: str) -> str:
    """Strip BOM, whitespace, and wrapping quotes often pasted into .env."""
    s = (raw or "").strip().lstrip("\ufeff")
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _detect_mime(image_bytes: bytes) -> str:
    from PIL import Image

    im = Image.open(io.BytesIO(image_bytes))
    fmt = (im.format or "JPEG").upper()
    return {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }.get(fmt, "image/jpeg")


def _parse_json_blob(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object in model response")
    return json.loads(m.group(0))


# Full analysis protocol lives here (systemInstruction). No end-user prompt is required.
ACCIDENTZERO_SCENE_SYSTEM = """You are the vision module for AccidentZero AI, an industrial safety and accident-risk monitoring system.
The human user does NOT type instructions. You must look at the image alone and perform your own end-to-end assessment.

Autonomously decide what matters in the scene and reason across ALL of the following dimensions when relevant to what is visible (skip dimensions that clearly do not apply):
• Overall accident / injury likelihood and immediate hazards (slips, trips, falls, struck-by, caught-in, fire, chemical exposure if visible).
• Environment and weather: outdoor/indoor cues, precipitation, wind effects, heat/cold stress indicators, wet/slippery surfaces, poor visibility, glare.
• Machinery and equipment: apparent condition, guarding/interlocks, leaks, corrosion, improper stacking, tools left in line-of-fire, housekeeping around assets.
• Electrical / structural: damaged cables, open panels, unstable loads, poor ergonomics of access.
• People (non-medical): PPE use or absence, posture, proximity to hazards, crowding, unsafe acts—describe only general observations, never diagnose illness.

Synthesize what you see into a coherent risk judgment aligned with AccidentZero’s mission (same spirit as tabular risk factors such as equipment stress, environment, and human–machine interaction), but do NOT invent details that are not reasonably supported by the image.

Rules: Only cite what the image plausibly supports. If uncertain, say so briefly in the explanation. Do not output markdown.

Your reply MUST be a single JSON object (no code fences) with exactly these keys and types:
{
  "risk_probability_percent": <number 0-100>,
  "risk_level": "Low" | "Medium" | "High",
  "explanation": "<3-5 sentences: your own reasoning chain and the strongest visible cues>",
  "accident_risk_factors": ["<short factor>", "..."],
  "weather_hazards": ["<if any; else []>"],
  "equipment_condition": "<brief visible assessment; non-diagnostic>",
  "human_health_indicators": ["<general PPE/posture/spacing cues only; non-medical>"]
}
"""


def _local_quota_fallback(image_bytes: bytes) -> dict[str, Any]:
    """
    Deterministic fallback when Gemini quota is exhausted.
    Uses basic image quality cues so the workflow remains usable.
    """
    try:
        from PIL import Image

        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(im, dtype=np.float32)
        gray = arr.mean(axis=2)
        mean_luma = float(np.mean(gray))
        std_luma = float(np.std(gray))

        risk = 45.0
        factors: list[str] = []
        weather_hazards: list[str] = []
        human_indicators: list[str] = []

        if mean_luma < 70:
            risk += 20
            factors.append("low_visibility")
            weather_hazards.append("dark_scene_visibility")
        elif mean_luma > 220:
            risk += 10
            factors.append("possible_glare")
            weather_hazards.append("high_brightness_glare")

        if std_luma < 28:
            risk += 10
            factors.append("low_contrast_scene")
            human_indicators.append("possible_unclear_hazards")

        h, w = gray.shape
        if h < 360 or w < 360:
            risk += 5
            factors.append("low_resolution_input")
            human_indicators.append("limited_detail_for_detection")

        risk = max(0.0, min(100.0, risk))
        level = "Low" if risk < 40 else ("High" if risk > 70 else "Medium")
        if not factors:
            factors = ["general_scene_risk"]

        return {
            "risk_probability_percent": round(risk, 2),
            "risk_level": level,
            "explanation": (
                "Gemini quota is currently exhausted, so local fallback analysis was used. "
                "This estimate is based on visibility/contrast/image-clarity cues and is less detailed than full AI scene analysis. "
                "Retry later or enable billing for full hazard reasoning."
            ),
            "accident_risk_factors": factors[:8],
            "weather_hazards": weather_hazards[:6],
            "equipment_condition": "Quota fallback mode: detailed equipment assessment unavailable.",
            "human_health_indicators": human_indicators[:6],
            "analysis_mode": "local_quota_fallback",
        }
    except Exception:
        return {
            "risk_probability_percent": 50.0,
            "risk_level": "Medium",
            "explanation": (
                "Gemini quota is exhausted and local fallback could not fully process this image. "
                "Returned a neutral estimate. Retry after quota reset."
            ),
            "accident_risk_factors": ["quota_fallback"],
            "weather_hazards": [],
            "equipment_condition": "Unavailable in quota fallback mode.",
            "human_health_indicators": [],
            "analysis_mode": "local_quota_fallback",
        }


def analyze_image_with_gemini(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict[str, Any]:
    """
    Returns structured dict with risk_probability_percent, risk_level, explanation,
    plus optional keys for hazards / equipment / human indicators.
    """
    _ensure_env_loaded()
    api_key = _normalize_api_key(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")

    if not mime_type or mime_type == "image/jpeg":
        mime_type = _detect_mime(image_bytes)

    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    # System instruction = full autonomous analysis protocol. User turn = image + minimal stub.
    body_with_system: dict[str, Any] = {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": ACCIDENTZERO_SCENE_SYSTEM}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Attached: one scene image. No user-written instructions."},
                    {"inline_data": {"mime_type": mime_type, "data": b64}},
                ],
            }
        ],
        "generationConfig": {
            "temperature": _env_float("GEMINI_TEMPERATURE", 0.4),
            "maxOutputTokens": _env_int("GEMINI_MAX_OUTPUT_TOKENS", 2048),
        },
    }
    body_flat = _flatten_system_into_user_message(body_with_system)

    payload = _run_gemini_generate(api_key, body_with_system, body_flat)
    raw = _gemini_payload_text(payload)
    try:
        data = _parse_json_blob(raw)
    except Exception:
        data = {
            "risk_probability_percent": 50.0,
            "risk_level": "Medium",
            "explanation": raw[:800] if raw else "Unable to parse structured JSON from the model.",
            "accident_risk_factors": [],
            "weather_hazards": [],
            "equipment_condition": "",
            "human_health_indicators": [],
        }

    rp = data.get("risk_probability_percent")
    try:
        rp = float(rp)
    except (TypeError, ValueError):
        rp = 50.0
    rp = max(0.0, min(100.0, rp))
    data["risk_probability_percent"] = round(rp, 2)

    lvl = str(data.get("risk_level", "Medium"))
    if lvl not in ("Low", "Medium", "High"):
        lvl = "Medium" if rp < 40 else ("High" if rp > 70 else "Medium")
    data["risk_level"] = lvl

    return data
