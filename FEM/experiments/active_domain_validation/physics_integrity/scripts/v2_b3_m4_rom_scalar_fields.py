#!/usr/bin/env python3
"""Phase-2 ROM scalar field definitions, alignment, and comparison metrics."""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PHASE2_PREDICTION_METHOD = "knn_idw_modal_surrogate_v2"
PHASE2_1_PREDICTION_METHOD = "knn_idw_modal_surrogate_v2_1"
MODEL_VERSION_V2_1 = "m4_modal_surrogate_v2_1_intensity"

ROM_PREDICTION_SCHEMA_V2 = "m4_rom_prediction_v2"
ROM_PREDICTION_SCHEMA_V2_1 = "m4_rom_prediction_v2_1"

ROM_TRAINING_CATALOG_SOURCE = "raw_modes_catalog_jsonl_deduped_for_rom"
ROM_COMPARE_CATALOG_SOURCE = "raw_modes_catalog_jsonl_deduped_for_rom"
ROM_DEDUPE_TOLERANCE_HZ = 0.05
INTENSITY_LOG_EPSILON = 1.0e-12
NORMALIZATION_PERCENTILE = 95
TOP_K_FRACTION = 0.2
ACCURACY_BAND_HZ_DEFAULT: Tuple[float, float] = (60.0, 550.0)

INTENSITY_FIELDS: Tuple[str, ...] = (
    "radiation_proxy",
    "mic_output_proxy",
    "bridge_excitation_abs",
)

INTENSITY_DERIVED_FIELDS: Tuple[str, ...] = tuple(
    f"{base}_{suffix}"
    for base in INTENSITY_FIELDS
    for suffix in ("log10", "p95_norm")
)

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


def _in_band(f_hz: float, band: Tuple[float, float]) -> bool:
    lo, hi = band
    return lo <= f_hz <= hi


def _percentile(vals: Sequence[float], pct: float) -> Optional[float]:
    if not vals:
        return None
    ordered = sorted(float(v) for v in vals)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(pct) / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def compute_intensity_p95_map(
    modes: Sequence[Mapping[str, Any]],
    *,
    band: Tuple[float, float] = ACCURACY_BAND_HZ_DEFAULT,
    percentile: int = NORMALIZATION_PERCENTILE,
) -> Dict[str, Optional[float]]:
    buckets: Dict[str, List[float]] = {f: [] for f in INTENSITY_FIELDS}
    for m in modes:
        f_hz = _safe_float(m.get("frequency_hz"))
        if f_hz is None or not _in_band(f_hz, band):
            continue
        for field in INTENSITY_FIELDS:
            v = _safe_float(m.get(field))
            if v is not None and v >= 0.0:
                buckets[field].append(v)
    return {field: _percentile(vals, float(percentile)) for field, vals in buckets.items()}


def enrich_mode_intensity_derivatives(
    mode: Dict[str, Any],
    *,
    p95_map: Mapping[str, Optional[float]],
    epsilon: float = INTENSITY_LOG_EPSILON,
) -> Dict[str, Any]:
    for field in INTENSITY_FIELDS:
        v = _safe_float(mode.get(field))
        if v is not None:
            mode[f"{field}_log10"] = round(math.log10(v + epsilon), 8)
        else:
            mode[f"{field}_log10"] = None
        p95 = p95_map.get(field)
        if v is not None and p95 is not None and p95 > 0.0:
            mode[f"{field}_p95_norm"] = round(v / p95, 8)
        else:
            mode[f"{field}_p95_norm"] = None
    return mode


def enrich_catalog_intensity_derivatives(
    modes: Sequence[Mapping[str, Any]],
    *,
    band: Tuple[float, float] = ACCURACY_BAND_HZ_DEFAULT,
    epsilon: float = INTENSITY_LOG_EPSILON,
    percentile: int = NORMALIZATION_PERCENTILE,
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[float]]]:
    p95_map = compute_intensity_p95_map(modes, band=band, percentile=percentile)
    out: List[Dict[str, Any]] = []
    for m in modes:
        rec = dict(m)
        enrich_mode_intensity_derivatives(rec, p95_map=p95_map, epsilon=epsilon)
        out.append(rec)
    return out, p95_map


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


def append_intensity_derivatives_to_prediction(
    mode_rec: Dict[str, Any],
    *,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
    target_hz: float,
    epsilon: float = INTENSITY_LOG_EPSILON,
) -> Dict[str, Any]:
    """Blend log10 and p95_norm intensity targets from enriched neighbor catalogs."""
    for field in INTENSITY_FIELDS:
        for suffix in ("log10", "p95_norm"):
            key = f"{field}_{suffix}"
            pairs: List[Tuple[float, float]] = []
            for catalog, wt in zip(neighbor_catalogs, neighbor_weights):
                idx = nearest_freq_mode_index(catalog, target_hz)
                if idx is None:
                    continue
                val = _safe_float(catalog[idx].get(key))
                if val is not None:
                    pairs.append((val, wt))
            val, _ = blend_numeric_field(pairs)
            if val is not None:
                mode_rec[key] = round(val, 8)
            else:
                raw = _safe_float(mode_rec.get(field))
                if suffix == "log10" and raw is not None:
                    mode_rec[key] = round(math.log10(raw + epsilon), 8)
                else:
                    mode_rec[key] = None
    return mode_rec


def _rank_correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None

    def _ranks(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx = _ranks(list(xs))
    ry = _ranks(list(ys))
    mx = statistics.mean(rx)
    my = statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return round(num / (den_x * den_y), 6)


def _top_k_overlap(
    fom_vals: Sequence[float],
    rom_vals: Sequence[float],
    *,
    fraction: float = TOP_K_FRACTION,
) -> Optional[float]:
    n = len(fom_vals)
    if n < 2:
        return None
    k = max(1, int(math.ceil(n * fraction)))
    fom_top = set(sorted(range(n), key=lambda i: fom_vals[i], reverse=True)[:k])
    rom_top = set(sorted(range(n), key=lambda i: rom_vals[i], reverse=True)[:k])
    return round(len(fom_top & rom_top) / float(k), 4)


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
    log_mae: Dict[str, List[float]] = {f: [] for f in INTENSITY_FIELDS}
    norm_mae: Dict[str, List[float]] = {f: [] for f in INTENSITY_FIELDS}
    rank_pairs: Dict[str, Tuple[List[float], List[float]]] = {f: ([], []) for f in INTENSITY_FIELDS}
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

            rom_log = _safe_float(m.get(f"rom_{field}_log10"))
            fom_log = _safe_float(m.get(f"fom_{field}_log10"))
            if rom_log is not None and fom_log is not None:
                log_mae[field].append(abs(rom_log - fom_log))

            rom_norm = _safe_float(m.get(f"rom_{field}_p95_norm"))
            fom_norm = _safe_float(m.get(f"fom_{field}_p95_norm"))
            if rom_norm is not None and fom_norm is not None:
                norm_mae[field].append(abs(rom_norm - fom_norm))

            if rom_v is not None and fom_v is not None:
                rank_pairs[field][0].append(fom_v)
                rank_pairs[field][1].append(rom_v)

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

    intensity_v21: Dict[str, Any] = {
        "intensity_log_epsilon": INTENSITY_LOG_EPSILON,
        "normalization_percentile": NORMALIZATION_PERCENTILE,
        "top_k_fraction": TOP_K_FRACTION,
        "radiation_proxy_log_mae": _mae(log_mae["radiation_proxy"]),
        "mic_output_proxy_log_mae": _mae(log_mae["mic_output_proxy"]),
        "bridge_excitation_abs_log_mae": _mae(log_mae["bridge_excitation_abs"]),
        "radiation_proxy_p95_norm_mae": _mae(norm_mae["radiation_proxy"]),
        "mic_output_proxy_p95_norm_mae": _mae(norm_mae["mic_output_proxy"]),
        "bridge_excitation_abs_p95_norm_mae": _mae(norm_mae["bridge_excitation_abs"]),
        "radiation_proxy_rank_correlation": _rank_correlation(
            *rank_pairs["radiation_proxy"]
        ),
        "mic_output_proxy_rank_correlation": _rank_correlation(*rank_pairs["mic_output_proxy"]),
        "bridge_excitation_rank_correlation": _rank_correlation(
            *rank_pairs["bridge_excitation_abs"]
        ),
        "top_k_radiation_overlap": _top_k_overlap(*rank_pairs["radiation_proxy"]),
        "top_k_mic_overlap": _top_k_overlap(*rank_pairs["mic_output_proxy"]),
        "top_k_bridge_overlap": _top_k_overlap(*rank_pairs["bridge_excitation_abs"]),
        "diagnostic_radiation_proxy_relative_error_median": _median(rad_rel),
        "diagnostic_mic_output_proxy_relative_error_median": _median(mic_rel),
        "diagnostic_bridge_excitation_abs_relative_error_median": _median(bridge_rel),
    }

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
            **intensity_v21,
        },
        "phase2_intensity_metrics_v2_1": intensity_v21,
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
    for field in INTENSITY_DERIVED_FIELDS:
        out[f"rom_{field}"] = _safe_float(rom_mode.get(field))
        out[f"fom_{field}"] = _safe_float(fom_mode.get(field))
    return out
