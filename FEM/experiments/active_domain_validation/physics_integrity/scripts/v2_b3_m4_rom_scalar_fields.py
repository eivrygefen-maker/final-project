#!/usr/bin/env python3
"""Phase-2 ROM scalar field definitions, alignment, and comparison metrics."""
from __future__ import annotations

import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PHASE2_PREDICTION_METHOD = "knn_idw_modal_surrogate_v2"
ROM_PREDICTION_SCHEMA_V2 = "m4_rom_prediction_v2"

PHASE2_REQUIRED_NUMERIC: Tuple[str, ...] = (
    "top_share",
    "back_share",
    "air_share",
    "bridge_excitation_coupling",
    "bridge_excitation_abs",
    "radiation_proxy",
    "mic_output_proxy",
    "modal_norm",
)

PHASE2_OPTIONAL_NUMERIC: Tuple[str, ...] = (
    "top_output_proxy",
    "back_output_proxy",
    "air_pressure_proxy",
)

PHASE2_REQUIRED_CATEGORICAL: Tuple[str, ...] = (
    "coupling_class",
    "dominant_region",
    "secondary_region",
)

PHASE2_OPTIONAL_CATEGORICAL: Tuple[str, ...] = (
    "bridge_excitation_region",
    "mic_output_method",
    "audio_coupling_status",
)

PHASE2_NUMERIC_FIELDS: Tuple[str, ...] = PHASE2_REQUIRED_NUMERIC + PHASE2_OPTIONAL_NUMERIC
PHASE2_CATEGORICAL_FIELDS: Tuple[str, ...] = PHASE2_REQUIRED_CATEGORICAL + PHASE2_OPTIONAL_CATEGORICAL

COUPLING_CLASS_VOCAB: Tuple[str, ...] = (
    "top_back_mixed",
    "back_dominant",
    "top_dominant",
    "air_dominant",
    "weak_or_unknown",
    "",
)

REGION_VOCAB: Tuple[str, ...] = ("top", "back", "air", "unknown", "")

# Soft targets for tracking (do not fail ROM on these yet)
SHARE_MAE_TARGET = 0.10
RADIATION_PROXY_REL_ERROR_TARGET = 0.25
MIC_OUTPUT_PROXY_REL_ERROR_TARGET = 0.25
COUPLING_CLASS_ACCURACY_TARGET = 0.70


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def extract_mode_scalars(mode: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"frequency_hz": _safe_float(mode.get("frequency_hz"))}
    for key in PHASE2_NUMERIC_FIELDS:
        out[key] = _safe_float(mode.get(key))
    for key in PHASE2_CATEGORICAL_FIELDS:
        raw = mode.get(key)
        out[key] = str(raw) if raw is not None else None
    return out


def nearest_freq_mode_index(modes: Sequence[Mapping[str, Any]], target_hz: float) -> Optional[int]:
    best_i: Optional[int] = None
    best_d = float("inf")
    for i, m in enumerate(modes):
        f = _safe_float(m.get("frequency_hz"))
        if f is None:
            continue
        d = abs(f - target_hz)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _encode_categorical(value: Any, vocab: Sequence[str]) -> int:
    s = str(value or "").strip()
    try:
        return int(vocab.index(s))
    except ValueError:
        return int(vocab.index("")) if "" in vocab else 0


def decode_categorical(idx: int, vocab: Sequence[str]) -> Optional[str]:
    if idx < 0 or idx >= len(vocab):
        return None
    s = vocab[idx]
    return s if s else None


def categorical_vocab_for_field(field: str) -> Tuple[str, ...]:
    if field == "coupling_class":
        return COUPLING_CLASS_VOCAB
    if field in ("dominant_region", "secondary_region"):
        return REGION_VOCAB
    return ("",)


def blend_numeric_field(
    values: Sequence[Tuple[float, float]],
) -> Tuple[Optional[float], float]:
    """Weighted average; returns (value, confidence as effective weight fraction)."""
    pairs = [(float(v), float(w)) for v, w in values if v == v and w > 0]
    if not pairs:
        return None, 0.0
    num = sum(v * w for v, w in pairs)
    den = sum(w for _, w in pairs)
    if den <= 0:
        return None, 0.0
    conf = min(1.0, den)
    return num / den, conf


def vote_categorical_field(
    values: Sequence[Tuple[str, float]],
) -> Tuple[Optional[str], float]:
    scores: Dict[str, float] = {}
    total = 0.0
    for val, wt in values:
        if not val:
            continue
        key = str(val)
        scores[key] = scores.get(key, 0.0) + float(wt)
        total += float(wt)
    if not scores or total <= 0:
        return None, 0.0
    winner = max(scores.items(), key=lambda kv: kv[1])
    return winner[0], min(1.0, winner[1] / total)


def predict_mode_scalars_at_frequency(
    *,
    target_hz: float,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
) -> Dict[str, Any]:
    """Nearest-frequency alignment within each neighbor, then IDW blend (Phase-2 default)."""
    rec: Dict[str, Any] = {"frequency_hz": round(float(target_hz), 6)}
    for field in PHASE2_NUMERIC_FIELDS:
        pairs: List[Tuple[float, float]] = []
        for catalog, wt in zip(neighbor_catalogs, neighbor_weights):
            idx = nearest_freq_mode_index(catalog, target_hz)
            if idx is None:
                continue
            val = _safe_float(catalog[idx].get(field))
            if val is not None:
                pairs.append((val, wt))
        val, conf = blend_numeric_field(pairs)
        rec[field] = round(val, 8) if val is not None else None
        if field in PHASE2_REQUIRED_NUMERIC:
            rec[f"{field}_confidence"] = round(conf, 4)

    confs = [rec.get(f"{f}_confidence") for f in ("top_share", "radiation_proxy", "modal_norm") if rec.get(f"{f}_confidence") is not None]
    rec["prediction_confidence"] = round(sum(confs) / len(confs), 4) if confs else 0.0

    for field in PHASE2_CATEGORICAL_FIELDS:
        pairs_cat: List[Tuple[str, float]] = []
        for catalog, wt in zip(neighbor_catalogs, neighbor_weights):
            idx = nearest_freq_mode_index(catalog, target_hz)
            if idx is None:
                continue
            raw = catalog[idx].get(field)
            if raw is not None:
                pairs_cat.append((str(raw), wt))
        val, conf = vote_categorical_field(pairs_cat)
        rec[field] = val
        if field in PHASE2_REQUIRED_CATEGORICAL:
            rec[f"{field}_confidence"] = round(conf, 4)

    rec["prediction_status"] = "computed"
    return rec


def _rel_err(rom: Optional[float], fom: Optional[float]) -> Optional[float]:
    if rom is None or fom is None or fom is None or abs(fom) < 1e-12:
        return None
    return abs(rom - fom) / abs(fom)


def _mae(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.mean(vals), 8) if vals else None


def _median(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.median(vals), 8) if vals else None


def compute_phase2_scalar_metrics(
    matches: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate STK/audio scalar and classification metrics on frequency-matched pairs."""
    top_share_errs: List[float] = []
    back_share_errs: List[float] = []
    air_share_errs: List[float] = []
    rad_rel: List[float] = []
    mic_rel: List[float] = []
    bridge_rel: List[float] = []
    coupling_hits = 0
    coupling_total = 0
    dom_hits = 0
    dom_total = 0
    audio_weighted_freq: List[Tuple[float, float]] = []
    audio_weighted_rad: List[Tuple[float, float]] = []

    for m in matches:
        rom_rad = _safe_float(m.get("rom_radiation_proxy") or m.get("fom_radiation_proxy"))
        fom_rad = _safe_float(m.get("fom_radiation_proxy"))
        wt = float(fom_rad) if fom_rad is not None and fom_rad > 0 else 1.0

        for field, bucket in (
            ("top_share", top_share_errs),
            ("back_share", back_share_errs),
            ("air_share", air_share_errs),
        ):
            rom_v = _safe_float(m.get(f"rom_{field}"))
            fom_v = _safe_float(m.get(f"fom_{field}"))
            if rom_v is not None and fom_v is not None:
                bucket.append(abs(rom_v - fom_v))

        for field, bucket in (
            ("radiation_proxy", rad_rel),
            ("mic_output_proxy", mic_rel),
            ("bridge_excitation_abs", bridge_rel),
        ):
            rom_v = _safe_float(m.get(f"rom_{field}"))
            fom_v = _safe_float(m.get(f"fom_{field}"))
            re = _rel_err(rom_v, fom_v)
            if re is not None:
                bucket.append(re)
                if field == "radiation_proxy":
                    audio_weighted_rad.append((re, wt))

        abs_hz = _safe_float(m.get("abs_error_hz"))
        rel_hz = _safe_float(m.get("relative_error"))
        if rel_hz is not None:
            audio_weighted_freq.append((rel_hz, wt))

        rom_cc = m.get("rom_coupling_class")
        fom_cc = m.get("fom_coupling_class")
        if rom_cc is not None and fom_cc is not None:
            coupling_total += 1
            if str(rom_cc) == str(fom_cc):
                coupling_hits += 1

        rom_dr = m.get("rom_dominant_region")
        fom_dr = m.get("fom_dominant_region")
        if rom_dr is not None and fom_dr is not None:
            dom_total += 1
            if str(rom_dr) == str(fom_dr):
                dom_hits += 1

    def _wmean(pairs: Sequence[Tuple[float, float]]) -> Optional[float]:
        if not pairs:
            return None
        num = sum(v * w for v, w in pairs)
        den = sum(w for _, w in pairs)
        return round(num / den, 8) if den > 0 else None

    return {
        "phase2_scalar_metrics": {
            "top_share_mae": _mae(top_share_errs),
            "back_share_mae": _mae(back_share_errs),
            "air_share_mae": _mae(air_share_errs),
            "radiation_proxy_relative_error_median": _median(rad_rel),
            "radiation_proxy_relative_error_mean": _mae(rad_rel),
            "mic_output_proxy_relative_error_median": _median(mic_rel),
            "mic_output_proxy_relative_error_mean": _mae(mic_rel),
            "bridge_excitation_abs_relative_error_median": _median(bridge_rel),
            "bridge_excitation_abs_relative_error_mean": _mae(bridge_rel),
            "coupling_class_accuracy": round(coupling_hits / coupling_total, 4) if coupling_total else None,
            "coupling_class_matched_count": coupling_total,
            "dominant_region_accuracy": round(dom_hits / dom_total, 4) if dom_total else None,
            "dominant_region_matched_count": dom_total,
            "audio_weighted_frequency_error": _wmean(audio_weighted_freq),
            "audio_weighted_output_proxy_error": _wmean(audio_weighted_rad),
        },
        "phase2_targets": {
            "frequency_median_relative_error_target": 0.05,
            "share_mae_target": SHARE_MAE_TARGET,
            "radiation_proxy_error_target": RADIATION_PROXY_REL_ERROR_TARGET,
            "mic_output_proxy_error_target": MIC_OUTPUT_PROXY_REL_ERROR_TARGET,
            "coupling_class_accuracy_target": COUPLING_CLASS_ACCURACY_TARGET,
        },
    }


def enrich_match_with_phase2_fields(
    match: Dict[str, Any],
    *,
    rom_mode: Mapping[str, Any],
    fom_mode: Mapping[str, Any],
) -> Dict[str, Any]:
    out = dict(match)
    for field in PHASE2_NUMERIC_FIELDS:
        out[f"rom_{field}"] = _safe_float(rom_mode.get(field))
        out[f"fom_{field}"] = _safe_float(fom_mode.get(field))
    for field in PHASE2_CATEGORICAL_FIELDS:
        out[f"rom_{field}"] = rom_mode.get(field)
        out[f"fom_{field}"] = fom_mode.get(field)
    return out
