#!/usr/bin/env python3
"""ROM Intensity v2.2 experimental — physics-aware matching, raw-blend normalization, ranking metrics."""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from v2_b3_m4_modal_surrogate_lib import (  # noqa: E402
    GEOMETRY_KEYS,
    _neighbor_catalog_from_arrays,
    _select_neighbors,
    predict_modal_catalog,
)
from v2_b3_m4_rom_scalar_fields import (  # noqa: E402
    ACCURACY_BAND_HZ_DEFAULT,
    INTENSITY_FIELDS,
    INTENSITY_LOG_EPSILON,
    NORMALIZATION_PERCENTILE,
    PHASE2_CATEGORICAL_FIELDS,
    PHASE2_NUMERIC_FIELDS,
    _rank_correlation,
    _safe_float,
    blend_numeric_field,
    compute_intensity_p95_map,
    enrich_mode_intensity_derivatives,
    predict_mode_scalars_at_frequency,
    vote_categorical_field,
)

MODEL_VERSION_V2_2 = "m4_modal_surrogate_v2_2_intensity_experimental"
PREDICTION_METHOD_V2_2 = "frequency_v21_plus_physics_aware_intensity_v22"
SURROGATE_SCHEMA_V2_2 = "m4_modal_surrogate_v2_2_experimental"
ROM_PREDICTION_SCHEMA_V2_2 = "m4_rom_prediction_v2_2_experimental"

INTENSITY_METHOD_V21_A = "v21_a_nearest_frequency_idw"
INTENSITY_METHOD_V22_B = "v22_b_physics_aware_idw"
INTENSITY_METHOD_V22_C = "v22_c_physics_aware_geometry_idw"
INTENSITY_METHOD_V22_D = "v22_d_physics_aware_ridge"

FREQ_BANDS_V22: Tuple[Tuple[float, float], ...] = (
    (60.0, 150.0),
    (150.0, 300.0),
    (300.0, 425.0),
    (425.0, 550.0),
)
TOP_K_FRACTIONS = (0.10, 0.20, 0.30)
MATCH_LEVEL_NAMES = (
    "class_region_band",
    "region_band",
    "class_band",
    "share_vector_frequency",
    "frequency_only",
)
MIN_RIDGE_SAMPLES = 12
RIDGE_ALPHA = 1.0


def band_for_frequency(f_hz: float) -> Tuple[float, float]:
    for lo, hi in FREQ_BANDS_V22:
        if lo <= f_hz <= hi:
            return (lo, hi)
    return ACCURACY_BAND_HZ_DEFAULT


def band_label(f_hz: float) -> str:
    lo, hi = band_for_frequency(f_hz)
    return f"{int(lo)}_{int(hi)}"


def derived_geometry_features(parameters: Mapping[str, Any]) -> Dict[str, float]:
    L = float(parameters.get("geometry.length") or 0.0)
    W = float(parameters.get("geometry.width") or 0.0)
    D = float(parameters.get("geometry.depth") or 0.0)
    r = float(parameters.get("geometry.hole_radius") or 0.0)
    tt = float(parameters.get("geometry.top_thickness") or 0.0)
    bt = float(parameters.get("geometry.back_thickness") or 0.0)
    hole_area = math.pi * r * r
    cavity = L * W * D
    return {
        "length": L,
        "width": W,
        "depth": D,
        "hole_radius": r,
        "top_thickness": tt,
        "back_thickness": bt,
        "hole_area": hole_area,
        "cavity_volume": cavity,
        "hole_area_over_cavity": hole_area / cavity if cavity > 1e-12 else 0.0,
        "length_over_width": L / W if W > 1e-12 else 0.0,
        "depth_over_length": D / L if L > 1e-12 else 0.0,
    }


def _geometry_feature_vector(parameters: Mapping[str, Any]) -> np.ndarray:
    g = derived_geometry_features(parameters)
    return np.array(
        [
            g["length"],
            g["width"],
            g["depth"],
            g["hole_radius"],
            g["hole_area"],
            g["cavity_volume"],
            g["hole_area_over_cavity"],
            g["length_over_width"],
            g["depth_over_length"],
            g["top_thickness"],
            g["back_thickness"],
        ],
        dtype=np.float64,
    )


def _encode_class_region(coupling_class: Any, dominant_region: Any) -> Tuple[int, int]:
    classes = ("top_back_mixed", "back_dominant", "top_dominant", "air_dominant", "weak_or_unknown", "")
    regions = ("top", "back", "air", "unknown", "")
    cc = str(coupling_class or "")
    dr = str(dominant_region or "")
    ci = classes.index(cc) if cc in classes else len(classes) - 1
    ri = regions.index(dr) if dr in regions else len(regions) - 1
    return ci, ri


def _modal_match_cost(
    mode: Mapping[str, Any],
    *,
    target_hz: float,
    target_class: Optional[str],
    target_region: Optional[str],
    target_shares: Tuple[Optional[float], Optional[float], Optional[float]],
    geometry_penalty: float = 0.0,
) -> float:
    f_hz = _safe_float(mode.get("frequency_hz")) or target_hz
    cost = abs(f_hz - target_hz) / 50.0
    tband = band_for_frequency(target_hz)
    mband = band_for_frequency(f_hz)
    if tband != mband:
        cost += 6.0
    if target_class and str(mode.get("coupling_class") or "") != str(target_class):
        cost += 8.0
    if target_region and str(mode.get("dominant_region") or "") != str(target_region):
        cost += 3.0
    for field, tgt in zip(("top_share", "back_share", "air_share"), target_shares):
        a = _safe_float(mode.get(field))
        if a is not None and tgt is not None:
            cost += abs(a - tgt) * 2.0
    cost += geometry_penalty
    return cost


def _matching_level(
    mode: Mapping[str, Any],
    *,
    target_hz: float,
    target_class: Optional[str],
    target_region: Optional[str],
) -> int:
    same_band = band_for_frequency(_safe_float(mode.get("frequency_hz")) or 0.0) == band_for_frequency(target_hz)
    if not same_band:
        return 5
    cc = str(mode.get("coupling_class") or "") == str(target_class or "")
    dr = str(mode.get("dominant_region") or "") == str(target_region or "")
    if cc and dr:
        return 1
    if dr:
        return 2
    if cc:
        return 3
    return 4


def select_neighbor_mode_physics_aware(
    catalog: Sequence[Mapping[str, Any]],
    *,
    target_hz: float,
    target_class: Optional[str],
    target_region: Optional[str],
    target_shares: Tuple[Optional[float], Optional[float], Optional[float]],
    geometry_penalty: float = 0.0,
) -> Tuple[Optional[int], Dict[str, Any]]:
    best_i: Optional[int] = None
    best_level = 99
    best_cost = float("inf")
    meta: Dict[str, Any] = {}
    for i, mode in enumerate(catalog):
        level = _matching_level(
            mode,
            target_hz=target_hz,
            target_class=target_class,
            target_region=target_region,
        )
        cost = _modal_match_cost(
            mode,
            target_hz=target_hz,
            target_class=target_class,
            target_region=target_region,
            target_shares=target_shares,
            geometry_penalty=geometry_penalty,
        )
        if level < best_level or (level == best_level and cost < best_cost):
            best_level = level
            best_cost = cost
            best_i = i
    if best_i is None:
        return None, {"matching_level_used": None, "fallback_used": True, "modal_match_cost": None}
    nm = catalog[best_i]
    level_name = MATCH_LEVEL_NAMES[min(best_level, len(MATCH_LEVEL_NAMES)) - 1] if best_level <= 5 else "frequency_only"
    meta = {
        "matching_level_used": level_name,
        "matched_neighbor_mode_frequency": _safe_float(nm.get("frequency_hz")),
        "matched_neighbor_coupling_class": nm.get("coupling_class"),
        "matched_neighbor_dominant_region": nm.get("dominant_region"),
        "modal_match_cost": round(best_cost, 6),
        "fallback_used": best_level >= 5,
    }
    return best_i, meta


def _predict_frequencies_only(
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    nev: int = 0,
) -> Dict[str, Any]:
    """Sorted-index IDW frequencies — identical to v2.1 frequency path."""
    arrays = model["arrays"]
    nn_idx, weights, neighbor_ids, dists = _select_neighbors(model, parameters)
    neighbor_freqs: List[np.ndarray] = []
    for j in nn_idx:
        count = int(arrays["mode_counts"][j])
        freqs = np.asarray(arrays["frequencies"][j, :count], dtype=np.float64)
        neighbor_freqs.append(freqs)
    min_count = min(len(f) for f in neighbor_freqs)
    mode_out = min(int(nev), min_count) if nev > 0 else min_count
    pred_freqs = np.zeros(mode_out, dtype=np.float64)
    for m in range(mode_out):
        pred_freqs[m] = np.average([float(f[m]) for f in neighbor_freqs], weights=weights)
    return {
        "frequencies_hz": [round(float(f), 6) for f in pred_freqs.tolist()],
        "nev_returned": int(mode_out),
        "k_neighbors_used": int(len(nn_idx)),
        "neighbor_sample_ids": neighbor_ids,
        "neighbor_distances": [round(float(dists[i]), 6) for i in nn_idx],
        "nn_idx": nn_idx,
        "weights": weights,
        "neighbor_freqs": neighbor_freqs,
    }


def _neighbor_catalogs(model: Mapping[str, Any], nn_idx: np.ndarray) -> List[List[Dict[str, Any]]]:
    arrays = model["arrays"]
    out: List[List[Dict[str, Any]]] = []
    for j in nn_idx:
        count = int(arrays["mode_counts"][j])
        out.append(_neighbor_catalog_from_arrays(arrays, int(j), count))
    return out


def _apply_post_hoc_p95_normalization(
    modes: List[Dict[str, Any]],
    *,
    epsilon: float = INTENSITY_LOG_EPSILON,
    percentile: int = NORMALIZATION_PERCENTILE,
    band: Tuple[float, float] = ACCURACY_BAND_HZ_DEFAULT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[float]]]:
    p95_map = compute_intensity_p95_map(modes, band=band, percentile=percentile)
    out: List[Dict[str, Any]] = []
    for m in modes:
        rec = dict(m)
        enrich_mode_intensity_derivatives(rec, p95_map=p95_map, epsilon=epsilon)
        for field in INTENSITY_FIELDS:
            rec[f"{field}_normalization_p95"] = p95_map.get(field)
        rec["intensity_log_epsilon"] = epsilon
        rec["normalization_percentile"] = percentile
        out.append(rec)
    return out, p95_map


def _blend_intensity_raw(
    *,
    field: str,
    target_hz: float,
    target_class: Optional[str],
    target_region: Optional[str],
    target_shares: Tuple[Optional[float], Optional[float], Optional[float]],
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
    method: str,
    geometry_penalty_by_neighbor: Optional[Sequence[float]] = None,
) -> Tuple[Optional[float], Dict[str, Any]]:
    pairs: List[Tuple[float, float]] = []
    match_meta: Dict[str, Any] = {}
    for ni, (catalog, wt) in enumerate(zip(neighbor_catalogs, neighbor_weights)):
        gpen = float(geometry_penalty_by_neighbor[ni]) if geometry_penalty_by_neighbor else 0.0
        if method == INTENSITY_METHOD_V21_A:
            from v2_b3_m4_rom_scalar_fields import nearest_freq_mode_index  # noqa: WPS433

            idx = nearest_freq_mode_index(catalog, target_hz)
            meta = {"matching_level_used": "frequency_only", "fallback_used": True}
        else:
            idx, meta = select_neighbor_mode_physics_aware(
                catalog,
                target_hz=target_hz,
                target_class=target_class,
                target_region=target_region,
                target_shares=target_shares,
                geometry_penalty=gpen,
            )
        if idx is None:
            continue
        val = _safe_float(catalog[idx].get(field))
        if val is not None:
            pairs.append((val, wt))
        if not match_meta:
            match_meta = meta
    val, _ = blend_numeric_field(pairs)
    return (round(val, 8) if val is not None else None), match_meta


def _fit_ridge(X: np.ndarray, y: np.ndarray, *, alpha: float = RIDGE_ALPHA) -> Optional[np.ndarray]:
    if len(y) < MIN_RIDGE_SAMPLES or X.shape[0] != len(y):
        return None
    X1 = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
    n_feat = X1.shape[1]
    try:
        w = np.linalg.solve(X1.T @ X1 + alpha * np.eye(n_feat), X1.T @ y)
    except np.linalg.LinAlgError:
        return None
    return w


def _predict_ridge(w: np.ndarray, x: np.ndarray) -> float:
    x1 = np.concatenate([[1.0], x])
    return float(x1 @ w)


def _mode_feature_row(
    parameters: Mapping[str, Any],
    mode: Mapping[str, Any],
    *,
    include_geometry: bool = True,
) -> np.ndarray:
    g = _geometry_feature_vector(parameters) if include_geometry else np.zeros(11)
    f_hz = _safe_float(mode.get("frequency_hz")) or 0.0
    lo, hi = band_for_frequency(f_hz)
    band_mid = (lo + hi) / 2.0
    ci, ri = _encode_class_region(mode.get("coupling_class"), mode.get("dominant_region"))
    return np.array(
        [
            f_hz / 550.0,
            band_mid / 550.0,
            _safe_float(mode.get("top_share")) or 0.0,
            _safe_float(mode.get("back_share")) or 0.0,
            _safe_float(mode.get("air_share")) or 0.0,
            float(ci) / 5.0,
            float(ri) / 4.0,
            *g.tolist(),
        ],
        dtype=np.float64,
    )


def _build_ridge_training(
    training_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    coupling_class: Optional[str] = None,
    band: Optional[Tuple[float, float]] = None,
    include_geometry: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    xs: List[np.ndarray] = []
    ys: List[float] = []
    for row in training_rows:
        params = dict(row.get("parameters") or {})
        for mode in row.get("mode_catalog") or []:
            f_hz = _safe_float(mode.get("frequency_hz"))
            if f_hz is None or not (ACCURACY_BAND_HZ_DEFAULT[0] <= f_hz <= ACCURACY_BAND_HZ_DEFAULT[1]):
                continue
            if coupling_class and str(mode.get("coupling_class") or "") != coupling_class:
                continue
            if band and not (band[0] <= f_hz <= band[1]):
                continue
            v = _safe_float(mode.get(field))
            if v is None:
                continue
            xs.append(_mode_feature_row(params, mode, include_geometry=include_geometry))
            ys.append(float(v))
    if len(xs) < MIN_RIDGE_SAMPLES:
        return None, None, None, None, "insufficient"
    X = np.vstack(xs)
    y = np.asarray(ys, dtype=np.float64)
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig = np.where(sig < 1e-12, 1.0, sig)
    return (X - mu) / sig, y, mu, sig, "ok"


def _fit_field_regressor_with_fallback(
    training_rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    pred_class: Optional[str],
    pred_band: Tuple[float, float],
    include_geometry: bool,
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, str]:
    strategies = [
        ("class_band", pred_class, pred_band),
        ("class_only", pred_class, None),
        ("band_only", None, pred_band),
        ("global", None, None),
    ]
    for name, cc, band in strategies:
        Xn, y, mu, sig, status = _build_ridge_training(
            training_rows,
            field=field,
            coupling_class=cc,
            band=band,
            include_geometry=include_geometry,
        )
        if status != "ok" or Xn is None or y is None or mu is None or sig is None:
            continue
        w = _fit_ridge(Xn, y)
        if w is not None:
            return w, mu, sig, name
    return None, np.array([]), np.array([]), "global_failed"


def predict_intensity_catalog_v22(
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    intensity_method: str = INTENSITY_METHOD_V22_B,
    training_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    nev: int = 0,
    excluded_sample_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    v2.2 prediction: v2.1 frequencies + experimental intensity path.
    training_rows required for method D (fitted only on training fold).
    """
    excluded = {str(s) for s in (excluded_sample_ids or model.get("excluded_sample_ids") or []) if s}
    freq_out = _predict_frequencies_only(model, parameters, nev=nev)
    nn_idx = freq_out["nn_idx"]
    weights = freq_out["weights"]
    neighbor_catalogs = _neighbor_catalogs(model, nn_idx)
    neighbor_ids = list(freq_out["neighbor_sample_ids"])
    if excluded and any(nid in excluded for nid in neighbor_ids):
        raise AssertionError(
            f"leakage: excluded sample in neighbors: {sorted(excluded & set(neighbor_ids))}"
        )

    # Shares + classification via existing nearest-frequency IDW (unchanged from v2.1)
    predicted_modes: List[Dict[str, Any]] = []
    match_stats: List[Dict[str, Any]] = []
    ridge_fallbacks: List[str] = []

    geometry_penalties: Optional[List[float]] = None
    if intensity_method == INTENSITY_METHOD_V22_C:
        tgt_g = _geometry_feature_vector(parameters)
        geometry_penalties = []
        arrays = model["arrays"]
        for j in nn_idx:
            # neighbor index row — use training sample parameters if available
            samples = list(model.get("training_samples") or [])
            sid = str(samples[int(j)].get("sample_id") or "") if int(j) < len(samples) else ""
            nparams = parameters
            if training_rows and sid:
                for tr in training_rows:
                    if str(tr.get("sample_id")) == sid:
                        nparams = dict(tr.get("parameters") or {})
                        break
            ng = _geometry_feature_vector(nparams)
            geometry_penalties.append(float(np.linalg.norm(tgt_g - ng)))

    ridge_weights: Dict[str, Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, str]] = {}

    for mi, f_hz in enumerate(freq_out["frequencies_hz"]):
        base = predict_mode_scalars_at_frequency(
            target_hz=float(f_hz),
            neighbor_catalogs=neighbor_catalogs,
            neighbor_weights=list(weights),
        )
        target_class = base.get("coupling_class")
        target_region = base.get("dominant_region")
        target_shares = (
            _safe_float(base.get("top_share")),
            _safe_float(base.get("back_share")),
            _safe_float(base.get("air_share")),
        )
        mode_rec = dict(base)
        mode_rec["frequency_hz"] = float(f_hz)
        mode_rec["mode_index"] = int(mi)
        mode_rec["frequency_band"] = band_label(float(f_hz))

        per_field_meta: Dict[str, Any] = {}
        for field in INTENSITY_FIELDS:
            raw: Optional[float] = None
            meta: Dict[str, Any] = {}
            if intensity_method == INTENSITY_METHOD_V22_D:
                if training_rows is None:
                    raise ValueError("method D requires training_rows from training fold only")
                key = (field, str(target_class), band_label(float(f_hz)))
                if key not in ridge_weights:
                    ridge_weights[key] = _fit_field_regressor_with_fallback(
                        training_rows,
                        field=field,
                        pred_class=str(target_class) if target_class else None,
                        pred_band=band_for_frequency(float(f_hz)),
                        include_geometry=True,
                    )
                w, mu, sig, fb = ridge_weights[key]
                ridge_fallbacks.append(fb)
                if w is not None and len(mu):
                    feat_mode = dict(base)
                    feat_mode["frequency_hz"] = f_hz
                    feat_mode["coupling_class"] = target_class
                    feat_mode["dominant_region"] = target_region
                    x = _mode_feature_row(parameters, feat_mode, include_geometry=True)
                    xn = (x - mu) / sig
                    raw = round(max(0.0, _predict_ridge(w, xn)), 8)
                    meta = {"regression_fallback": fb, "matching_level_used": "ridge_regression", "fallback_used": False}
            if raw is None:
                blend_method = (
                    INTENSITY_METHOD_V22_B
                    if intensity_method == INTENSITY_METHOD_V22_D
                    else intensity_method
                )
                raw, meta = _blend_intensity_raw(
                    field=field,
                    target_hz=float(f_hz),
                    target_class=str(target_class) if target_class else None,
                    target_region=str(target_region) if target_region else None,
                    target_shares=target_shares,
                    neighbor_catalogs=neighbor_catalogs,
                    neighbor_weights=list(weights),
                    method=blend_method,
                    geometry_penalty_by_neighbor=geometry_penalties,
                )
            mode_rec[field] = raw
            per_field_meta[field] = meta
            for k, v in meta.items():
                mode_rec[f"{field}_{k}"] = v

        match_stats.append(per_field_meta.get("mic_output_proxy") or {})

        predicted_modes.append(mode_rec)

    predicted_modes, p95_map = _apply_post_hoc_p95_normalization(predicted_modes)

    # Top-k membership scores (rank by raw within predicted guitar)
    for field in INTENSITY_FIELDS:
        vals = [_safe_float(m.get(field)) or 0.0 for m in predicted_modes]
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
        n = len(vals)
        for frac in TOP_K_FRACTIONS:
            k = max(1, int(math.ceil(n * frac)))
            top_set = set(order[:k])
            for i, m in enumerate(predicted_modes):
                m[f"{field}_top_{int(frac*100)}pct_member"] = bool(i in top_set)

    fallback_rate = None
    if match_stats:
        fb = sum(1 for m in predicted_modes if m.get("mic_output_proxy_fallback_used"))
        fallback_rate = round(fb / len(predicted_modes), 4)

    return {
        "frequencies_hz": freq_out["frequencies_hz"],
        "predicted_modes": predicted_modes,
        "nev_returned": freq_out["nev_returned"],
        "k_neighbors_used": freq_out["k_neighbors_used"],
        "neighbor_sample_ids": neighbor_ids,
        "neighbor_distances": freq_out["neighbor_distances"],
        "method": PREDICTION_METHOD_V2_2,
        "intensity_method": intensity_method,
        "model_version": MODEL_VERSION_V2_2,
        "intensity_normalization": "raw_blend_then_predicted_guitar_p95",
        "intensity_p95_map": p95_map,
        "intensity_log_epsilon": INTENSITY_LOG_EPSILON,
        "normalization_percentile": NORMALIZATION_PERCENTILE,
        "frequency_alignment": "sorted_index_idw",
        "scalar_alignment": "physics_aware_v22",
        "fallback_rate": fallback_rate,
        "ridge_regression_fallbacks": ridge_fallbacks,
        "training_includes_target": False,
        "excluded_sample_ids": list(excluded),
    }


def predict_v21_baseline(model: Mapping[str, Any], parameters: Mapping[str, Any], *, nev: int = 0) -> Dict[str, Any]:
    """Method A — delegate to production v2.1 path."""
    out = predict_modal_catalog(model, parameters, nev=nev)
    out["intensity_method"] = INTENSITY_METHOD_V21_A
    return out


def _kendall_tau(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    try:
        from scipy.stats import kendalltau  # noqa: WPS433

        if len(xs) < 3:
            return None
        tau, _ = kendalltau(xs, ys)
        return round(float(tau), 6) if tau == tau else None
    except Exception:
        return None


def _ndcg_at_fraction(fom_vals: Sequence[float], rom_vals: Sequence[float], *, fraction: float) -> Optional[float]:
    n = len(fom_vals)
    if n < 2:
        return None
    k = max(1, int(math.ceil(n * fraction)))
    fom_order = sorted(range(n), key=lambda i: fom_vals[i], reverse=True)
    rom_order = sorted(range(n), key=lambda i: rom_vals[i], reverse=True)[:k]
    dcg = sum((2.0 ** fom_vals[i] - 1.0) / math.log2(rank + 2.0) for rank, i in enumerate(rom_order))
    ideal = sorted(fom_vals, reverse=True)[:k]
    idcg = sum((2.0 ** v - 1.0) / math.log2(rank + 2.0) for rank, v in enumerate(ideal))
    if idcg <= 0:
        return None
    return round(dcg / idcg, 6)


def _topk_metrics(
    fom_vals: Sequence[float],
    rom_vals: Sequence[float],
    *,
    fraction: float,
) -> Dict[str, Optional[float]]:
    n = len(fom_vals)
    if n < 2:
        return {"overlap": None, "precision": None, "recall": None}
    k = max(1, int(math.ceil(n * fraction)))
    fom_top = set(sorted(range(n), key=lambda i: fom_vals[i], reverse=True)[:k])
    rom_top = set(sorted(range(n), key=lambda i: rom_vals[i], reverse=True)[:k])
    inter = fom_top & rom_top
    prec = len(inter) / float(len(rom_top)) if rom_top else None
    rec = len(inter) / float(len(fom_top)) if fom_top else None
    return {
        "overlap": round(len(inter) / float(k), 4),
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
    }


def compute_intensity_metrics_v22(
    matches: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Extended v2.2 metrics including multi-fraction top-k, Kendall, NDCG."""
    from v2_b3_m4_rom_scalar_fields import compute_phase2_scalar_metrics  # noqa: E402

    base = compute_phase2_scalar_metrics(matches)
    p2 = dict(base.get("phase2_scalar_metrics") or {})
    rank_pairs: Dict[str, Tuple[List[float], List[float]]] = {f: ([], []) for f in INTENSITY_FIELDS}
    norm_mae: Dict[str, List[float]] = {f: [] for f in INTENSITY_FIELDS}

    for m in matches:
        for field in INTENSITY_FIELDS:
            rom_v = _safe_float(m.get(f"rom_{field}"))
            fom_v = _safe_float(m.get(f"fom_{field}"))
            rom_n = _safe_float(m.get(f"rom_{field}_p95_norm"))
            fom_n = _safe_float(m.get(f"fom_{field}_p95_norm"))
            if rom_n is not None and fom_n is not None:
                norm_mae[field].append(abs(rom_n - fom_n))
            if rom_v is not None and fom_v is not None:
                rank_pairs[field][0].append(fom_v)
                rank_pairs[field][1].append(rom_v)

    extra: Dict[str, Any] = {}
    for field in INTENSITY_FIELDS:
        fom_xs, rom_ys = rank_pairs[field]
        extra[f"{field}_spearman"] = _rank_correlation(fom_xs, rom_ys)
        extra[f"{field}_kendall_tau"] = _kendall_tau(fom_xs, rom_ys)
        for frac in TOP_K_FRACTIONS:
            pct = int(frac * 100)
            tk = _topk_metrics(fom_xs, rom_ys, fraction=frac)
            extra[f"{field}_top_{pct}pct_overlap"] = tk["overlap"]
            extra[f"{field}_top_{pct}pct_precision"] = tk["precision"]
            extra[f"{field}_top_{pct}pct_recall"] = tk["recall"]
            extra[f"{field}_ndcg_at_{pct}pct"] = _ndcg_at_fraction(fom_xs, rom_ys, fraction=frac)
        extra[f"{field}_p95_norm_mae"] = round(statistics.mean(norm_mae[field]), 8) if norm_mae[field] else None

    p2.update(extra)
    p2["metrics_schema"] = "m4_rom_intensity_metrics_v22"
    return {"phase2_scalar_metrics": p2, **base}
