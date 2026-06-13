#!/usr/bin/env python3
"""Stage 5.2 diagnostic-only audio proxy prediction candidates (no production overwrite)."""
from __future__ import annotations

import copy
import math
import statistics
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_rom_scalar_fields import (  # noqa: E402
    ACCURACY_BAND_HZ_DEFAULT,
    INTENSITY_FIELDS,
    INTENSITY_LOG_EPSILON,
    TOP_K_FRACTION,
    blend_numeric_field,
    compute_phase2_scalar_metrics,
    nearest_freq_mode_index,
    predict_mode_scalars_at_frequency,
    vote_categorical_field,
)

CandidateFn = Callable[..., List[Dict[str, Any]]]

CANDIDATE_A = "candidate_a_log_target_calibration"
CANDIDATE_B = "candidate_b_rank_preserving_local_calibration"
CANDIDATE_C = "candidate_c_top_k_audio_mode_classifier"
CANDIDATE_D = "candidate_d_hybrid_proxy_prediction"

ALL_CANDIDATES: Tuple[str, ...] = (
    CANDIDATE_A,
    CANDIDATE_B,
    CANDIDATE_C,
    CANDIDATE_D,
)


def _freq_band(f_hz: float) -> str:
    if f_hz < 150.0:
        return "low"
    if f_hz < 300.0:
        return "mid"
    return "high"


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _percentile(vals: Sequence[float], pct: float) -> float:
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


def _neighbor_intensity_pool(
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    field: str,
    *,
    band: Tuple[float, float] = ACCURACY_BAND_HZ_DEFAULT,
) -> List[float]:
    lo, hi = band
    out: List[float] = []
    for catalog in neighbor_catalogs:
        for mode in catalog:
            f_hz = _safe_float(mode.get("frequency_hz"))
            if f_hz is None or not (lo <= f_hz <= hi):
                continue
            v = _safe_float(mode.get(field))
            if v is not None and v >= 0.0:
                out.append(v)
    return out


def apply_candidate_a_log_target_calibration(
    predicted_modes: Sequence[Mapping[str, Any]],
    *,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
    epsilon: float = INTENSITY_LOG_EPSILON,
) -> List[Dict[str, Any]]:
    """Blend log10 intensity targets, then inverse-transform to linear domain."""
    out: List[Dict[str, Any]] = [dict(m) for m in predicted_modes]
    for mode in out:
        target_hz = float(mode.get("frequency_hz") or 0.0)
        for field in INTENSITY_FIELDS:
            pairs: List[Tuple[float, float]] = []
            for catalog, wt in zip(neighbor_catalogs, neighbor_weights):
                idx = nearest_freq_mode_index(catalog, target_hz)
                if idx is None:
                    continue
                raw = _safe_float(catalog[idx].get(field))
                if raw is not None:
                    pairs.append((math.log10(raw + epsilon), float(wt)))
            val, _ = blend_numeric_field(pairs)
            if val is not None:
                mode[field] = round(10.0 ** float(val) - epsilon, 8)
    return out


def apply_candidate_b_rank_preserving_local_calibration(
    predicted_modes: Sequence[Mapping[str, Any]],
    *,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
    blend: float = 0.55,
) -> List[Dict[str, Any]]:
    """Calibrate intensity distribution using neighbor percentiles while preserving rank order."""
    out: List[Dict[str, Any]] = [dict(m) for m in predicted_modes]
    for field in INTENSITY_FIELDS:
        pool = _neighbor_intensity_pool(neighbor_catalogs, field)
        if len(pool) < 3:
            continue
        pool.sort()
        ranked = sorted(
            [(i, m) for i, m in enumerate(out) if _safe_float(m.get(field)) is not None],
            key=lambda t: float(t[1][field]),
        )
        n = len(ranked)
        if n == 0:
            continue
        for rank_i, (idx, mode) in enumerate(ranked):
            frac = rank_i / max(n - 1, 1)
            target = _percentile(pool, frac * 100.0)
            orig = float(mode[field])
            mode[field] = round((1.0 - blend) * orig + blend * target, 8)
    return out


def apply_candidate_c_top_k_audio_mode_classifier(
    predicted_modes: Sequence[Mapping[str, Any]],
    *,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
    top_fraction: float = TOP_K_FRACTION,
) -> List[Dict[str, Any]]:
    """Add rank/classifier fields; reweight radiation_proxy for top audio modes only."""
    out: List[Dict[str, Any]] = [dict(m) for m in predicted_modes]
    rad_vals = [_safe_float(m.get("radiation_proxy")) or 0.0 for m in out]
    n = len(out)
    k = max(1, int(math.ceil(n * top_fraction)))
    top_idx = set(sorted(range(n), key=lambda i: rad_vals[i], reverse=True)[:k])

    for i, mode in enumerate(out):
        rad = rad_vals[i]
        if i in top_idx:
            mode["is_top_audio_mode"] = True
            mode["radiation_rank_bucket"] = "top"
            mode["radiation_proxy"] = round(rad * 1.15, 8)
        elif rad >= _percentile(rad_vals, 50.0):
            mode["is_top_audio_mode"] = False
            mode["radiation_rank_bucket"] = "mid"
        else:
            mode["is_top_audio_mode"] = False
            mode["radiation_rank_bucket"] = "low"
            mode["radiation_proxy"] = round(rad * 0.85, 8)

        pairs: List[Tuple[str, float]] = []
        target_hz = float(mode.get("frequency_hz") or 0.0)
        for catalog, wt in zip(neighbor_catalogs, neighbor_weights):
            idx = nearest_freq_mode_index(catalog, target_hz)
            if idx is None:
                continue
            dr = catalog[idx].get("dominant_region")
            if dr:
                pairs.append((str(dr), float(wt)))
        voted, _ = vote_categorical_field(pairs)
        if voted:
            mode["dominant_region"] = voted
    return out


def apply_candidate_d_hybrid_proxy_prediction(
    predicted_modes: Sequence[Mapping[str, Any]],
    *,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
    baseline_modes: Optional[Sequence[Mapping[str, Any]]] = None,
    epsilon: float = INTENSITY_LOG_EPSILON,
) -> List[Dict[str, Any]]:
    """Base prediction + region/band conditioning + neighbor residual correction."""
    base = [dict(m) for m in (baseline_modes or predicted_modes)]
    log_cal = apply_candidate_a_log_target_calibration(
        base,
        neighbor_catalogs=neighbor_catalogs,
        neighbor_weights=neighbor_weights,
        epsilon=epsilon,
    )
    out: List[Dict[str, Any]] = []
    for mi, mode in enumerate(log_cal):
        rec = dict(mode)
        target_hz = float(rec.get("frequency_hz") or 0.0)
        band = _freq_band(target_hz)
        region = str(rec.get("dominant_region") or "unknown")
        baseline_mode = (baseline_modes or log_cal)[mi] if mi < len(baseline_modes or log_cal) else rec

        for field in INTENSITY_FIELDS:
            residual_pairs: List[Tuple[float, float]] = []
            region_pairs: List[Tuple[float, float]] = []
            band_pairs: List[Tuple[float, float]] = []
            for catalog, wt in zip(neighbor_catalogs, neighbor_weights):
                idx = nearest_freq_mode_index(catalog, target_hz)
                if idx is None:
                    continue
                neighbor_mode = catalog[idx]
                n_val = _safe_float(neighbor_mode.get(field))
                if n_val is None:
                    continue
                baseline_at = _safe_float(baseline_mode.get(field))
                if baseline_at is not None:
                    residual_pairs.append((n_val - baseline_at, float(wt)))
                if str(neighbor_mode.get("dominant_region") or "") == region:
                    region_pairs.append((n_val, float(wt)))
                if _freq_band(float(neighbor_mode.get("frequency_hz") or 0.0)) == band:
                    band_pairs.append((n_val, float(wt)))

            current = _safe_float(rec.get(field))
            if current is None:
                continue
            corrected = current
            residual, _ = blend_numeric_field(residual_pairs)
            region_val, _ = blend_numeric_field(region_pairs)
            band_val, _ = blend_numeric_field(band_pairs)
            if residual is not None:
                corrected += 0.35 * residual
            if region_val is not None:
                corrected = 0.6 * corrected + 0.4 * region_val
            if band_val is not None:
                corrected = 0.75 * corrected + 0.25 * band_val
            rec[field] = round(max(corrected, 0.0), 8)
        out.append(rec)
    return out


def get_candidate_fn(name: str) -> CandidateFn:
    mapping: Dict[str, CandidateFn] = {
        CANDIDATE_A: apply_candidate_a_log_target_calibration,
        CANDIDATE_B: apply_candidate_b_rank_preserving_local_calibration,
        CANDIDATE_C: apply_candidate_c_top_k_audio_mode_classifier,
        CANDIDATE_D: apply_candidate_d_hybrid_proxy_prediction,
    }
    if name not in mapping:
        raise KeyError(f"unknown candidate: {name}")
    return mapping[name]


def apply_audio_proxy_candidate(
    name: str,
    predicted_modes: Sequence[Mapping[str, Any]],
    *,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
    baseline_modes: Optional[Sequence[Mapping[str, Any]]] = None,
    epsilon: float = INTENSITY_LOG_EPSILON,
) -> List[Dict[str, Any]]:
    fn = get_candidate_fn(name)
    kwargs: Dict[str, Any] = {
        "neighbor_catalogs": neighbor_catalogs,
        "neighbor_weights": neighbor_weights,
    }
    if name == CANDIDATE_D:
        kwargs["baseline_modes"] = baseline_modes or predicted_modes
        kwargs["epsilon"] = epsilon
    elif name == CANDIDATE_A:
        kwargs["epsilon"] = epsilon
    return fn(predicted_modes, **kwargs)


def rebuild_matches_with_rom_modes(
    comparison: Mapping[str, Any],
    rom_modes: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    from v2_b3_m4_rom_fom_compare_lib import _enrich_matches_with_phase2_scalars  # noqa: WPS433

    matches = copy.deepcopy(list(comparison.get("per_mode_matches") or []))
    fom_modes = []
    for m in matches:
        fom_modes.append(
            {
                "frequency_hz": m.get("fom_frequency_hz"),
                "coupling_class": m.get("fom_coupling_class"),
                "radiation_proxy": m.get("fom_radiation_proxy"),
                "mic_output_proxy": m.get("fom_mic_output_proxy"),
                "top_share": m.get("fom_top_share"),
                "back_share": m.get("fom_back_share"),
                "air_share": m.get("fom_air_share"),
                "dominant_region": m.get("fom_dominant_region"),
                "bridge_excitation_abs": m.get("fom_bridge_excitation_abs"),
            }
        )
    return _enrich_matches_with_phase2_scalars(
        matches,
        rom_modes=rom_modes,
        fom_modes=fom_modes,
    )


def evaluate_candidate_on_comparison(
    comparison: Mapping[str, Any],
    candidate_name: str,
    *,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
    baseline_modes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    baseline = list(baseline_modes or [])
    if not baseline:
        for m in comparison.get("per_mode_matches") or []:
            baseline.append(
                {
                    "frequency_hz": m.get("rom_frequency_hz"),
                    "radiation_proxy": m.get("rom_radiation_proxy"),
                    "mic_output_proxy": m.get("rom_mic_output_proxy"),
                    "dominant_region": m.get("rom_dominant_region"),
                    "coupling_class": m.get("rom_coupling_class"),
                }
            )
    adjusted = apply_audio_proxy_candidate(
        candidate_name,
        baseline,
        neighbor_catalogs=neighbor_catalogs,
        neighbor_weights=neighbor_weights,
        baseline_modes=baseline,
    )
    matches = rebuild_matches_with_rom_modes(comparison, adjusted)
    metrics = compute_phase2_scalar_metrics(matches)
    return {
        "candidate": candidate_name,
        "phase2_scalar_metrics": metrics.get("phase2_scalar_metrics") or {},
        "matched_mode_count": len(matches),
    }


def summarize_candidate_improvements(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    keys = (
        "radiation_proxy_log_mae",
        "mic_output_proxy_p95_norm_mae",
        "top_k_radiation_overlap",
        "radiation_proxy_rank_correlation",
        "coupling_class_accuracy",
        "dominant_region_accuracy",
    )

    def _delta(key: str) -> Optional[float]:
        b = baseline_metrics.get(key)
        c = candidate_metrics.get(key)
        if b is None or c is None:
            return None
        if key.endswith("_mae"):
            return round(float(b) - float(c), 8)
        return round(float(c) - float(b), 8)

    out: Dict[str, Any] = {}
    for key in keys:
        b = baseline_metrics.get(key)
        c = candidate_metrics.get(key)
        pct: Optional[float] = None
        if b is not None and c is not None and float(b) != 0.0 and key.endswith("_mae"):
            pct = round((float(b) - float(c)) / abs(float(b)), 4)
        out[key] = {
            "baseline": b,
            "candidate": c,
            "delta": _delta(key),
            "improvement_fraction": pct,
        }
    return out


def _fom_catalog_from_comparison(comparison: Mapping[str, Any]) -> List[Dict[str, Any]]:
    modes: List[Dict[str, Any]] = []
    for m in comparison.get("per_mode_matches") or []:
        modes.append(
            {
                "frequency_hz": m.get("fom_frequency_hz"),
                "coupling_class": m.get("fom_coupling_class"),
                "dominant_region": m.get("fom_dominant_region"),
                "secondary_region": m.get("fom_secondary_region"),
                "radiation_proxy": m.get("fom_radiation_proxy"),
                "mic_output_proxy": m.get("fom_mic_output_proxy"),
                "bridge_excitation_abs": m.get("fom_bridge_excitation_abs"),
                "top_share": m.get("fom_top_share"),
                "back_share": m.get("fom_back_share"),
                "air_share": m.get("fom_air_share"),
            }
        )
    return modes


def _baseline_rom_modes_from_comparison(comparison: Mapping[str, Any]) -> List[Dict[str, Any]]:
    modes: List[Dict[str, Any]] = []
    for m in comparison.get("per_mode_matches") or []:
        modes.append(
            {
                "frequency_hz": m.get("rom_frequency_hz"),
                "coupling_class": m.get("rom_coupling_class"),
                "dominant_region": m.get("rom_dominant_region"),
                "radiation_proxy": m.get("rom_radiation_proxy"),
                "mic_output_proxy": m.get("rom_mic_output_proxy"),
                "bridge_excitation_abs": m.get("rom_bridge_excitation_abs"),
                "top_share": m.get("rom_top_share"),
                "back_share": m.get("rom_back_share"),
                "air_share": m.get("rom_air_share"),
            }
        )
    return modes


def evaluate_candidates_on_comparisons(
    comparisons: Sequence[Mapping[str, Any]],
    *,
    candidate_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Diagnostic candidate sweep using per-comparison FOM catalog as neighbor pool."""
    names = list(candidate_names or ALL_CANDIDATES)
    baseline_samples: List[Dict[str, Any]] = []
    per_candidate: Dict[str, List[Dict[str, Any]]] = {n: [] for n in names}

    for comp in comparisons:
        sid = comp.get("sample_id")
        baseline_p2 = (comp.get("phase2_scalar_metrics") or {})
        baseline_samples.append({"sample_id": sid, "phase2_scalar_metrics": baseline_p2})
        fom_catalog = _fom_catalog_from_comparison(comp)
        if not fom_catalog:
            continue
        baseline_modes = _baseline_rom_modes_from_comparison(comp)
        neighbor_catalogs = [fom_catalog]
        neighbor_weights = [1.0]
        for name in names:
            result = evaluate_candidate_on_comparison(
                comp,
                name,
                neighbor_catalogs=neighbor_catalogs,
                neighbor_weights=neighbor_weights,
                baseline_modes=baseline_modes,
            )
            per_candidate[name].append(
                {
                    "sample_id": sid,
                    "phase2_scalar_metrics": result.get("phase2_scalar_metrics") or {},
                    "improvements_vs_baseline": summarize_candidate_improvements(
                        baseline_p2,
                        result.get("phase2_scalar_metrics") or {},
                    ),
                }
            )

    def _median_metric(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        vals = [
            float((r.get("phase2_scalar_metrics") or {}).get(key))
            for r in rows
            if (r.get("phase2_scalar_metrics") or {}).get(key) is not None
        ]
        return round(statistics.median(vals), 8) if vals else None

    summary: Dict[str, Any] = {}
    baseline_agg = {
        k: _median_metric(baseline_samples, k)
        for k in (
            "radiation_proxy_log_mae",
            "mic_output_proxy_p95_norm_mae",
            "top_k_radiation_overlap",
            "radiation_proxy_rank_correlation",
            "coupling_class_accuracy",
            "dominant_region_accuracy",
        )
    }
    for name in names:
        rows = per_candidate[name]
        cand_agg = {
            k: _median_metric(rows, k)
            for k in baseline_agg.keys()
        }
        summary[name] = {
            "baseline_median_metrics": baseline_agg,
            "candidate_median_metrics": cand_agg,
            "improvements_vs_baseline": summarize_candidate_improvements(baseline_agg, cand_agg),
            "per_sample": rows,
            "diagnostic_only": True,
            "neighbor_pool": "fom_catalog_from_same_comparison",
        }
    return {
        "baseline_median_metrics": baseline_agg,
        "candidates": summary,
    }


def diagnose_audio_proxy_weakness(
    comparisons: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Heuristic diagnosis from aggregated LOO per-mode matches."""
    all_rad: List[float] = []
    rad_errors: List[float] = []
    freq_errors: List[float] = []
    top_dom_errors = 0
    top_dom_total = 0
    by_band: Dict[str, List[float]] = {"low": [], "mid": [], "high": []}
    by_region: Dict[str, List[float]] = {}
    rank_buckets: Dict[str, List[float]] = {"top20": [], "mid": [], "background": []}

    for comp in comparisons:
        matches = comp.get("per_mode_matches") or []
        rad_fom = [
            _safe_float(m.get("fom_radiation_proxy")) or 0.0
            for m in matches
            if _safe_float(m.get("fom_radiation_proxy")) is not None
        ]
        if rad_fom:
            all_rad.extend(rad_fom)
        n = len(rad_fom)
        k = max(1, int(math.ceil(n * TOP_K_FRACTION)))
        top_indices = set(sorted(range(n), key=lambda i: rad_fom[i], reverse=True)[:k]) if n else set()

        for i, m in enumerate(matches):
            f_hz = _safe_float(m.get("fom_frequency_hz"))
            rom_rad = _safe_float(m.get("rom_radiation_proxy"))
            fom_rad = _safe_float(m.get("fom_radiation_proxy"))
            if fom_rad is not None and fom_rad > 0 and rom_rad is not None:
                rad_errors.append(abs(rom_rad - fom_rad) / fom_rad)
            rel_hz = _safe_float(m.get("relative_error"))
            if rel_hz is not None and f_hz is not None:
                freq_errors.append(rel_hz)
                by_band[_freq_band(f_hz)].append(rel_hz)
            if fom_rad is not None and rom_rad is not None:
                if i in top_indices:
                    rank_buckets["top20"].append(abs(rom_rad - fom_rad) / max(fom_rad, 1e-12))
                elif fom_rad >= statistics.median(rad_fom) if rad_fom else False:
                    rank_buckets["mid"].append(abs(rom_rad - fom_rad) / max(fom_rad, 1e-12))
                else:
                    rank_buckets["background"].append(abs(rom_rad - fom_rad) / max(fom_rad, 1e-12))
            dr = m.get("fom_dominant_region")
            if dr and rom_rad is not None and fom_rad is not None:
                by_region.setdefault(str(dr), []).append(abs(rom_rad - fom_rad) / max(fom_rad, 1e-12))
            if m.get("rom_dominant_region") is not None and m.get("fom_dominant_region") is not None:
                top_dom_total += 1
                if str(m.get("rom_dominant_region")) != str(m.get("fom_dominant_region")):
                    top_dom_errors += 1

    heavy_tail = False
    if len(all_rad) >= 5:
        p50 = _percentile(all_rad, 50.0)
        p95 = _percentile(all_rad, 95.0)
        heavy_tail = p95 > 5.0 * max(p50, 1e-12)

    sparse_fraction = None
    if all_rad:
        sparse_fraction = round(sum(1 for v in all_rad if v <= _percentile(all_rad, 10.0)) / len(all_rad), 4)

    top_share = None
    if all_rad:
        ordered = sorted(all_rad, reverse=True)
        k = max(1, int(math.ceil(len(ordered) * TOP_K_FRACTION)))
        top_share = round(sum(ordered[:k]) / max(sum(ordered), 1e-12), 4)

    knn_peak_loss = None
    if rad_errors:
        knn_peak_loss = round(statistics.mean(rad_errors), 6)

    freq_align_corr = None
    if len(freq_errors) >= 3 and len(rad_errors) >= 3:
        n = min(len(freq_errors), len(rad_errors))
        try:
            freq_align_corr = round(
                statistics.correlation(freq_errors[:n], rad_errors[:n]),
                4,
            )
        except statistics.StatisticsError:
            freq_align_corr = None

    rank_vs_abs = {}
    for bucket, errs in rank_buckets.items():
        rank_vs_abs[bucket] = round(statistics.mean(errs), 6) if errs else None

    hypotheses = []
    if heavy_tail:
        hypotheses.append("target_distribution_heavy_tailed")
    if sparse_fraction is not None and sparse_fraction > 0.5:
        hypotheses.append("proxies_sparse_in_tail")
    if top_share is not None and top_share > 0.65:
        hypotheses.append("top_modes_dominate_radiation_budget")
    if knn_peak_loss is not None and knn_peak_loss > 0.25:
        hypotheses.append("knn_idw_averages_away_peaks")
    if freq_align_corr is not None and abs(freq_align_corr) > 0.35:
        hypotheses.append("frequency_alignment_mismatch_affects_proxy_compare")

    absolute_poor = knn_peak_loss is not None and knn_peak_loss > 0.30
    top20_err = rank_vs_abs.get("top20")
    bg_err = rank_vs_abs.get("background")
    rank_better_than_abs = (
        top20_err is not None
        and bg_err is not None
        and top20_err < bg_err * 0.85
    )

    return {
        "heavy_tailed_target_distribution": heavy_tail,
        "sparse_proxy_fraction_bottom_decile": sparse_fraction,
        "top_modes_radiation_share": top_share,
        "mean_radiation_relative_error": knn_peak_loss,
        "frequency_vs_radiation_error_correlation": freq_align_corr,
        "dominant_region_mismatch_rate": round(top_dom_errors / top_dom_total, 4) if top_dom_total else None,
        "error_by_frequency_band_hz": {
            band: round(statistics.mean(errs), 6) if errs else None for band, errs in by_band.items()
        },
        "error_by_dominant_region": {
            region: round(statistics.mean(errs), 6) if errs else None for region, errs in by_region.items()
        },
        "error_by_audio_importance_rank": rank_vs_abs,
        "likely_contributors": hypotheses,
        "absolute_radiation_prediction_weak": absolute_poor,
        "rank_or_top_k_relatively_better": rank_better_than_abs,
        "both_absolute_and_rank_weak": absolute_poor and not rank_better_than_abs,
    }
