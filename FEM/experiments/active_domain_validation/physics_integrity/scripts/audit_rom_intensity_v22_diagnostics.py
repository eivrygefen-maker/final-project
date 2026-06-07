#!/usr/bin/env python3
"""Read-only ROM intensity diagnostics for v2.2 planning (no FOM/ROM production changes)."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    DEFAULT_RUN_ID_SUFFIX,
    is_lhs_entry_completed,
    load_lhs_pool,
)
from v2_b3_m4_modal_surrogate_lib import (  # noqa: E402
    DEFAULT_K_NEIGHBORS,
    build_surrogate_from_training_rows,
    collect_completed_fom_training_rows,
    encode_lhs_parameters,
    guitars_root,
    predict_modal_catalog,
)
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    ACCURACY_BAND_HZ,
    greedy_nearest_hz_match,
    load_fom_modes_catalog_deduped,
    prepare_fom_modes_for_rom_compare,
    resolve_sample_context,
)
from v2_b3_m4_rom_scalar_fields import (  # noqa: E402
    INTENSITY_FIELDS,
    TOP_K_FRACTION,
    blend_numeric_field,
    compute_phase2_scalar_metrics,
    enrich_match_with_phase2_fields,
    nearest_freq_mode_index,
    predict_mode_scalars_at_frequency,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402

FREQ_BANDS: Tuple[Tuple[float, float], ...] = (
    (60.0, 150.0),
    (150.0, 300.0),
    (300.0, 425.0),
    (425.0, 550.0),
)
LEARNING_POOL_SIZES = (8, 12, 16, 20, 24, 29)
LHS_CLOSE_THRESHOLDS = (0.5, 1.0, 1.5, 2.0)

_DIAGNOSTICS_API_CHECKED = False


def _check_diagnostics_api() -> None:
    """Fail fast before the long diagnostics loop if a required helper is missing."""
    global _DIAGNOSTICS_API_CHECKED
    if _DIAGNOSTICS_API_CHECKED:
        return
    required: Dict[str, Any] = {
        "resolve_sample_context": resolve_sample_context,
        "load_fom_modes_catalog_deduped": load_fom_modes_catalog_deduped,
        "prepare_fom_modes_for_rom_compare": prepare_fom_modes_for_rom_compare,
        "greedy_nearest_hz_match": greedy_nearest_hz_match,
        "predict_modal_catalog": predict_modal_catalog,
        "build_surrogate_from_training_rows": build_surrogate_from_training_rows,
        "collect_completed_fom_training_rows": collect_completed_fom_training_rows,
    }
    missing = [name for name, obj in required.items() if obj is None]
    if missing:
        raise ImportError(f"diagnostics API self-check failed; missing: {missing}")
    _DIAGNOSTICS_API_CHECKED = True


def _parse_sample_range(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _sample_id(i: int) -> str:
    return f"sample_{i:03d}"


def _median(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.median(vals), 6) if vals else None


def _mean(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.mean(vals), 6) if vals else None


def _band_label(f_hz: float) -> str:
    for lo, hi in FREQ_BANDS:
        if lo <= f_hz <= hi:
            return f"{int(lo)}_{int(hi)}"
    return "out_of_band"


def _lhs_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    xa = encode_lhs_parameters(a)
    xb = encode_lhs_parameters(b)
    return float(np.linalg.norm(xa - xb))


def _assert_training_subset(
    training_sample_ids: Sequence[str],
    eligible_sample_ids: Sequence[str],
    *,
    context: str,
) -> None:
    eligible = set(eligible_sample_ids)
    bad = [str(s) for s in training_sample_ids if str(s) not in eligible]
    if bad:
        raise AssertionError(
            f"{context}: training samples outside eligible universe: {bad[:5]}"
            f"{'...' if len(bad) > 5 else ''}"
        )


def _discover_eligible_samples(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    requested_sample_ids: Sequence[str],
    run_id_suffix: str = DEFAULT_RUN_ID_SUFFIX,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str]]:
    """Return eligible IDs = requested ∩ COMPLETED ∩ catalog exists."""
    entries_by_id: Dict[str, Mapping[str, Any]] = {}
    for entry in pool.get("entries") or []:
        sid = str(entry.get("id") or "").strip()
        if sid:
            entries_by_id[sid] = entry

    eligible_ids: List[str] = []
    eligible_entries: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}

    for sid in requested_sample_ids:
        entry = entries_by_id.get(sid)
        if entry is None:
            skipped[sid] = "not_in_lhs_pool"
            continue
        run_id, run_root = _resolve_run_location(
            repo_root=repo_root,
            sample_id=sid,
            entry=entry,
            run_id_suffix=run_id_suffix,
        )
        if not is_lhs_entry_completed(entry, run_id=run_id):
            skipped[sid] = f"status_not_completed:{entry.get('status')}"
            continue
        catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
        if not catalog_path.is_file():
            skipped[sid] = "missing_modes_catalog"
            continue
        eligible_ids.append(sid)
        eligible_entries.append(dict(entry))

    return eligible_ids, eligible_entries, skipped


def _select_nearest_training_rows(
    *,
    eligible_entries: Sequence[Mapping[str, Any]],
    eligible_sample_ids: Sequence[str],
    target_params: Mapping[str, Any],
    exclude_sample_id: str,
    max_count: int,
) -> List[str]:
    eligible = set(eligible_sample_ids)
    scored: List[Tuple[float, str]] = []
    for entry in eligible_entries:
        sid = str(entry.get("id") or "")
        if not sid or sid == exclude_sample_id or sid not in eligible:
            continue
        params = dict(entry.get("parameters") or {})
        scored.append((_lhs_distance(target_params, params), sid))
    scored.sort(key=lambda t: t[0])
    k = min(int(max_count), max(0, len(scored)))
    return [sid for _, sid in scored[:k]]


def _collect_rows_for_samples(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_ids: Sequence[str],
    eligible_sample_ids: Sequence[str],
    context: str,
) -> List[Dict[str, Any]]:
    _assert_training_subset(sample_ids, eligible_sample_ids, context=context)
    wanted = [str(s) for s in sample_ids]
    out: List[Dict[str, Any]] = []
    for sid in wanted:
        rows, skipped = collect_completed_fom_training_rows(
            repo_root=repo_root,
            pool=pool,
            completed_only=True,
            force_sample=sid,
            exclude_sample_ids=[],
            max_samples=1,
        )
        if not rows:
            reasons = [str(s.get("reason") or "") for s in skipped if s.get("sample_id") == sid]
            raise RuntimeError(
                f"{context}: missing completed catalog for {sid}"
                + (f" ({reasons[0]})" if reasons else "")
            )
        out.append(rows[0])
    return out


def _resolve_run_location(
    *,
    repo_root: Path,
    sample_id: str,
    entry: Mapping[str, Any],
    run_id_suffix: str = DEFAULT_RUN_ID_SUFFIX,
) -> Tuple[str, Path]:
    """Mirror run_m4_rom_compare / select_completed_lhs_for_rom_compare run resolution."""
    run_id = str(entry.get("last_run_id") or f"{sample_id}_{run_id_suffix}")
    last_run_dir = entry.get("last_run_dir")
    if isinstance(last_run_dir, str) and last_run_dir.strip():
        run_root = Path(last_run_dir.strip())
        if not run_root.is_absolute():
            run_root = repo_root / run_root
    else:
        run_root = guitars_root(repo_root) / sample_id / "runs" / run_id
    return run_id, run_root.resolve()


def _run_context_for_sample(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    entry: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    run_id, run_root = _resolve_run_location(
        repo_root=repo_root,
        sample_id=sample_id,
        entry=entry,
    )
    context = resolve_sample_context(
        pool=pool,
        sample_id=sample_id,
        run_id=run_id,
        run_root=run_root,
        repo_root=repo_root,
    )
    catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"missing FOM catalog: {catalog_path}")
    _raw, deduped, meta = load_fom_modes_catalog_deduped(catalog_path)
    fom_modes, _ = prepare_fom_modes_for_rom_compare(deduped, band=ACCURACY_BAND_HZ)
    context["fom_catalog_path"] = str(catalog_path)
    return context, fom_modes, meta


def _metrics_from_prediction(
    *,
    prediction: Mapping[str, Any],
    fom_modes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rom_freqs = [float(f) for f in (prediction.get("frequencies_hz") or [])]
    rom_modes = list(prediction.get("predicted_modes") or [])
    matches, _ = greedy_nearest_hz_match(rom_frequencies_hz=rom_freqs, fom_modes=fom_modes)
    enriched = []
    for m in matches:
        rom_hz = float(m.get("rom_frequency_hz") or 0.0)
        fom_hz = float(m.get("fom_frequency_hz") or 0.0)
        rom_mode = min(
            (dict(x) for x in rom_modes),
            key=lambda r: abs(float(r.get("frequency_hz") or 0.0) - rom_hz),
            default={},
        )
        fom_mode = min(
            (dict(x) for x in fom_modes),
            key=lambda r: abs(float(r.get("frequency_hz") or 0.0) - fom_hz),
            default={},
        )
        enriched.append(enrich_match_with_phase2_fields(m, rom_mode=rom_mode, fom_mode=fom_mode))
    freq_rel = [
        float(m["relative_error"])
        for m in enriched
        if m.get("relative_error") is not None and m["relative_error"] == m["relative_error"]
    ]
    phase2 = compute_phase2_scalar_metrics(enriched)
    p2 = phase2.get("phase2_scalar_metrics") or {}
    return {
        "matched_mode_count": len(enriched),
        "frequency_median_relative_error": _median(freq_rel),
        "mic_output_proxy_p95_norm_mae": p2.get("mic_output_proxy_p95_norm_mae"),
        "radiation_proxy_p95_norm_mae": p2.get("radiation_proxy_p95_norm_mae"),
        "bridge_excitation_abs_p95_norm_mae": p2.get("bridge_excitation_abs_p95_norm_mae"),
        "radiation_proxy_rank_correlation": p2.get("radiation_proxy_rank_correlation"),
        "mic_output_proxy_rank_correlation": p2.get("mic_output_proxy_rank_correlation"),
        "bridge_excitation_rank_correlation": p2.get("bridge_excitation_rank_correlation"),
        "top_k_radiation_overlap": p2.get("top_k_radiation_overlap"),
        "top_k_mic_overlap": p2.get("top_k_mic_overlap"),
        "top_k_bridge_overlap": p2.get("top_k_bridge_overlap"),
        "per_mode_matches": enriched,
    }


def _neighbor_stats(model: Mapping[str, Any], parameters: Mapping[str, Any]) -> Dict[str, float]:
    arrays = model["arrays"]
    x = encode_lhs_parameters(parameters)
    x_norm = (x - arrays["feature_mean"]) / arrays["feature_std"]
    dists = np.linalg.norm(arrays["feature_matrix_norm"] - x_norm.reshape(1, -1), axis=1)
    k = int(min(model.get("k_neighbors") or DEFAULT_K_NEIGHBORS, len(dists)))
    k = max(1, k)
    nn_idx = np.argsort(dists)[:k]
    nn_d = [float(dists[i]) for i in nn_idx]
    return {
        "nearest_training_distance": round(min(nn_d), 6),
        "mean_k_neighbor_distance": round(float(np.mean(nn_d)), 6),
        "k_neighbors_used": int(k),
    }


def _predict_v21(model: Mapping[str, Any], parameters: Mapping[str, Any]) -> Dict[str, Any]:
    return predict_modal_catalog(model, parameters, nev=0)


def _nearest_mode_index_class_aware(
    modes: Sequence[Mapping[str, Any]],
    target_hz: float,
    *,
    prefer_class: Optional[str] = None,
    prefer_region: Optional[str] = None,
    band: Optional[Tuple[float, float]] = None,
) -> Optional[int]:
    best_i: Optional[int] = None
    best_score = float("inf")
    for i, m in enumerate(modes):
        f = m.get("frequency_hz")
        try:
            f_hz = float(f)
        except (TypeError, ValueError):
            continue
        if band is not None and not (band[0] <= f_hz <= band[1]):
            continue
        d = abs(f_hz - target_hz)
        score = d
        if prefer_class and str(m.get("coupling_class") or "") != str(prefer_class):
            score += 8.0
        if prefer_region and str(m.get("dominant_region") or "") != str(prefer_region):
            score += 3.0
        if score < best_score:
            best_score = score
            best_i = i
    return best_i


def _predict_scalars_baseline(
    *,
    target_hz: float,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    neighbor_weights: Sequence[float],
    method: str,
    predicted_class: Optional[str] = None,
    predicted_region: Optional[str] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"frequency_hz": round(float(target_hz), 6)}
    band = None
    for lo, hi in FREQ_BANDS:
        if lo <= target_hz <= hi:
            band = (lo, hi)
            break

    for field in INTENSITY_FIELDS:
        pairs: List[Tuple[float, float]] = []
        for catalog, wt in zip(neighbor_catalogs, neighbor_weights):
            if method == "global_band_mean":
                vals = []
                for m in catalog:
                    f = m.get("frequency_hz")
                    try:
                        f_hz = float(f)
                    except (TypeError, ValueError):
                        continue
                    if band and not (band[0] <= f_hz <= band[1]):
                        continue
                    v = m.get(f"{field}_p95_norm")
                    if v is not None:
                        vals.append(float(v))
                if vals:
                    pairs.append((float(statistics.mean(vals)), wt))
                continue

            if method == "nearest_single":
                idx = nearest_freq_mode_index(catalog, target_hz)
            elif method == "class_aware":
                idx = _nearest_mode_index_class_aware(
                    catalog,
                    target_hz,
                    prefer_class=predicted_class,
                    prefer_region=predicted_region,
                )
            elif method == "class_band_aware":
                idx = _nearest_mode_index_class_aware(
                    catalog,
                    target_hz,
                    prefer_class=predicted_class,
                    prefer_region=predicted_region,
                    band=band,
                )
            else:
                idx = nearest_freq_mode_index(catalog, target_hz)
            if idx is None:
                continue
            v = m.get(f"{field}_p95_norm") if (m := catalog[idx]) else None
            if v is not None:
                pairs.append((float(v), wt))
        val, _ = blend_numeric_field(pairs)
        rec[f"{field}_p95_norm"] = round(val, 8) if val is not None else None
    return rec


def _predict_baseline_catalog(
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    *,
    method: str,
) -> Dict[str, Any]:
    base = predict_modal_catalog(model, parameters, nev=0)
    arrays = model["arrays"]
    x = encode_lhs_parameters(parameters)
    x_norm = (x - arrays["feature_mean"]) / arrays["feature_std"]
    dists = np.linalg.norm(arrays["feature_matrix_norm"] - x_norm.reshape(1, -1), axis=1)
    k = int(min(model.get("k_neighbors") or DEFAULT_K_NEIGHBORS, len(dists)))
    k = max(1, k)
    nn_idx = np.argsort(dists)[:k]
    weights = 1.0 / (dists[nn_idx] + 1e-8) ** 2
    weights = weights / weights.sum()

    neighbor_catalogs = []
    for j in nn_idx:
        count = int(arrays["mode_counts"][j])
        from v2_b3_m4_modal_surrogate_lib import _neighbor_catalog_from_arrays  # noqa: WPS433

        neighbor_catalogs.append(_neighbor_catalog_from_arrays(arrays, int(j), count))

    if method == "nearest_single":
        nn_idx = nn_idx[:1]
        weights = np.array([1.0])
        neighbor_catalogs = neighbor_catalogs[:1]

    modes_out = []
    for mi, f_hz in enumerate(base.get("frequencies_hz") or []):
        class_rec = predict_mode_scalars_at_frequency(
            target_hz=float(f_hz),
            neighbor_catalogs=neighbor_catalogs,
            neighbor_weights=list(weights),
        )
        scalar_rec = _predict_scalars_baseline(
            target_hz=float(f_hz),
            neighbor_catalogs=neighbor_catalogs,
            neighbor_weights=list(weights),
            method=method,
            predicted_class=class_rec.get("coupling_class"),
            predicted_region=class_rec.get("dominant_region"),
        )
        scalar_rec["mode_index"] = int(mi)
        scalar_rec["frequency_hz"] = round(float(f_hz), 6)
        scalar_rec["coupling_class"] = class_rec.get("coupling_class")
        scalar_rec["dominant_region"] = class_rec.get("dominant_region")
        modes_out.append(scalar_rec)

    return {
        "frequencies_hz": base.get("frequencies_hz"),
        "predicted_modes": modes_out,
    }


def _alignment_audit(
    *,
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    fom_modes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    pred = predict_modal_catalog(model, parameters, nev=0)
    arrays = model["arrays"]
    x = encode_lhs_parameters(parameters)
    x_norm = (x - arrays["feature_mean"]) / arrays["feature_std"]
    dists = np.linalg.norm(arrays["feature_matrix_norm"] - x_norm.reshape(1, -1), axis=1)
    k = int(min(model.get("k_neighbors") or DEFAULT_K_NEIGHBORS, len(dists)))
    nn_idx = np.argsort(dists)[: max(1, k)]
    weights = 1.0 / (dists[nn_idx] + 1e-8) ** 2
    weights = weights / weights.sum()
    from v2_b3_m4_modal_surrogate_lib import _neighbor_catalog_from_arrays  # noqa: WPS433

    neighbor_catalogs = [
        _neighbor_catalog_from_arrays(arrays, int(j), int(arrays["mode_counts"][j])) for j in nn_idx
    ]

    rows = []
    for fom in fom_modes:
        f_hz = float(fom["frequency_hz"])
        class_pairs = []
        region_pairs = []
        share_errs = []
        for catalog, wt in zip(neighbor_catalogs, weights):
            idx = nearest_freq_mode_index(catalog, f_hz)
            if idx is None:
                continue
            nm = catalog[idx]
            class_pairs.append((str(nm.get("coupling_class") or ""), str(fom.get("coupling_class") or ""), wt))
            region_pairs.append((str(nm.get("dominant_region") or ""), str(fom.get("dominant_region") or ""), wt))
            for field in ("top_share", "back_share", "air_share"):
                a = nm.get(field)
                b = fom.get(field)
                if a is not None and b is not None:
                    share_errs.append((abs(float(a) - float(b)), wt))
        class_match = (
            max(class_pairs, key=lambda t: t[2])[0] == max(class_pairs, key=lambda t: t[2])[1]
            if class_pairs
            else None
        )
        region_match = (
            max(region_pairs, key=lambda t: t[2])[0] == max(region_pairs, key=lambda t: t[2])[1]
            if region_pairs
            else None
        )
        share_mae = None
        if share_errs:
            num = sum(e * w for e, w in share_errs)
            den = sum(w for _, w in share_errs)
            share_mae = round(num / den, 6) if den > 0 else None
        rows.append(
            {
                "fom_frequency_hz": f_hz,
                "fom_coupling_class": fom.get("coupling_class"),
                "neighbor_class_match": class_match,
                "neighbor_region_match": region_match,
                "neighbor_share_mae": share_mae,
                "band": _band_label(f_hz),
            }
        )

    def _subset(name: str, pred) -> Dict[str, Any]:
        subset = [r for r in rows if pred(r)]
        if not subset:
            return {"count": 0}
        return {
            "count": len(subset),
            "class_match_rate": round(
                sum(1 for r in subset if r.get("neighbor_class_match")) / len(subset), 4
            ),
            "region_match_rate": round(
                sum(1 for r in subset if r.get("neighbor_region_match")) / len(subset), 4
            ),
            "mean_share_mae": _mean(
                [float(r["neighbor_share_mae"]) for r in subset if r.get("neighbor_share_mae") is not None]
            ),
        }

    return {
        "mode_count": len(rows),
        "overall_class_match_rate": round(
            sum(1 for r in rows if r.get("neighbor_class_match")) / len(rows), 4
        )
        if rows
        else None,
        "overall_region_match_rate": round(
            sum(1 for r in rows if r.get("neighbor_region_match")) / len(rows), 4
        )
        if rows
        else None,
        "by_coupling_class": {
            cc: _subset(cc, lambda r, cc=cc: r.get("fom_coupling_class") == cc)
            for cc in sorted({str(r.get("fom_coupling_class") or "") for r in rows})
        },
        "by_band": {
            band: _subset(band, lambda r, band=band: r.get("band") == band)
            for band in sorted({str(r.get("band") or "") for r in rows})
        },
    }


def _repeatability_from_entries(eligible_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    entries = [e for e in eligible_entries if e.get("parameters")]
    pairs = []
    for i, a in enumerate(entries):
        pa = dict(a.get("parameters") or {})
        for b in entries[i + 1 :]:
            pb = dict(b.get("parameters") or {})
            d = _lhs_distance(pa, pb)
            pairs.append((d, str(a.get("id")), str(b.get("id"))))
    pairs.sort(key=lambda t: t[0])

    # Noise floor estimate: not mode-level here (needs catalogs); report LHS geometry spread.
    dists = [p[0] for p in pairs]
    return {
        "pair_count": len(pairs),
        "lhs_distance_min": round(min(dists), 6) if dists else None,
        "lhs_distance_median": _median(dists),
        "lhs_distance_p10": round(float(np.percentile(dists, 10)), 6) if dists else None,
        "closest_pairs": [
            {"distance": round(d, 6), "sample_a": a, "sample_b": b} for d, a, b in pairs[:10]
        ],
    }


def _aggregate_metric(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Optional[float]]:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return {
        f"{key}_median": _median(vals),
        f"{key}_mean": _mean(vals),
        f"{key}_worst": round(max(vals), 6) if vals else None,
    }


def run_diagnostics(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    requested_sample_ids: Sequence[str],
    eligible_sample_ids: Sequence[str],
    eligible_entries: Sequence[Mapping[str, Any]],
    k_neighbors: int,
) -> Dict[str, Any]:
    if not eligible_sample_ids:
        raise RuntimeError("no eligible completed samples in requested range")

    entries_by_id = {str(e.get("id")): e for e in eligible_entries}
    max_train_available = max(0, len(eligible_sample_ids) - 1)

    def _context_for(sid: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        entry = entries_by_id.get(sid)
        if entry is None:
            raise KeyError(f"eligible entry missing for {sid}")
        return _run_context_for_sample(
            repo_root=repo_root,
            pool=pool,
            sample_id=sid,
            entry=entry,
        )

    learning_curve: Dict[str, Any] = {}
    for pool_size in LEARNING_POOL_SIZES:
        t0 = time.perf_counter()
        per_sample = []
        effective_pool_size = min(int(pool_size), max_train_available)
        for sid in eligible_sample_ids:
            entry = entries_by_id.get(sid)
            if entry is None:
                continue
            params = dict(entry.get("parameters") or {})
            train_ids = _select_nearest_training_rows(
                eligible_entries=eligible_entries,
                eligible_sample_ids=eligible_sample_ids,
                target_params=params,
                exclude_sample_id=sid,
                max_count=effective_pool_size,
            )
            _assert_training_subset(
                train_ids,
                eligible_sample_ids,
                context=f"learning_curve pool_size={pool_size} holdout={sid}",
            )
            train_rows = _collect_rows_for_samples(
                repo_root=repo_root,
                pool=pool,
                sample_ids=train_ids,
                eligible_sample_ids=eligible_sample_ids,
                context=f"learning_curve pool_size={pool_size} holdout={sid}",
            )
            model = build_surrogate_from_training_rows(
                shape_name=str(pool.get("shape_name") or "classic"),
                training_rows=train_rows,
                k_neighbors=min(k_neighbors, len(train_rows)),
            )
            context, fom_modes, _ = _context_for(sid)
            pred = _predict_v21(model, context["parameters"])
            metrics = _metrics_from_prediction(prediction=pred, fom_modes=fom_modes)
            metrics["sample_id"] = sid
            metrics["training_pool_size"] = len(train_ids)
            metrics["training_pool_size_requested"] = effective_pool_size
            per_sample.append(metrics)
        agg = {
            "training_pool_size": pool_size,
            "training_pool_size_effective": effective_pool_size,
            "sample_count": len(per_sample),
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        for key in (
            "frequency_median_relative_error",
            "mic_output_proxy_p95_norm_mae",
            "radiation_proxy_p95_norm_mae",
            "bridge_excitation_abs_p95_norm_mae",
            "radiation_proxy_rank_correlation",
            "mic_output_proxy_rank_correlation",
            "top_k_radiation_overlap",
            "top_k_mic_overlap",
        ):
            agg.update(_aggregate_metric(per_sample, key))
        learning_curve[str(pool_size)] = {"aggregate": agg, "per_sample": per_sample}

    holdout_rows = []
    distance_rows = []
    alignment_rows = []
    baseline_rows: Dict[str, List[Dict[str, Any]]] = {
        "v21_knn_idw": [],
        "nearest_single": [],
        "global_band_mean": [],
        "class_aware": [],
        "class_band_aware": [],
    }

    for sid in eligible_sample_ids:
        train_ids = [s for s in eligible_sample_ids if s != sid]
        _assert_training_subset(
            train_ids,
            eligible_sample_ids,
            context=f"full_holdout holdout={sid}",
        )
        train_rows = _collect_rows_for_samples(
            repo_root=repo_root,
            pool=pool,
            sample_ids=train_ids,
            eligible_sample_ids=eligible_sample_ids,
            context=f"full_holdout holdout={sid}",
        )
        model = build_surrogate_from_training_rows(
            shape_name=str(pool.get("shape_name") or "classic"),
            training_rows=train_rows,
            k_neighbors=min(k_neighbors, len(train_rows)),
        )
        context, fom_modes, _ = _context_for(sid)
        pred = _predict_v21(model, context["parameters"])
        metrics = _metrics_from_prediction(prediction=pred, fom_modes=fom_modes)
        nstats = _neighbor_stats(model, context["parameters"])
        row = {
            "sample_id": sid,
            **nstats,
            **{k: metrics.get(k) for k in metrics if k != "per_mode_matches"},
        }
        holdout_rows.append(row)
        distance_rows.append(row)
        alignment_rows.append(
            {
                "sample_id": sid,
                **_alignment_audit(model=model, parameters=context["parameters"], fom_modes=fom_modes),
            }
        )
        for method, bucket in baseline_rows.items():
            if method == "v21_knn_idw":
                pred_m = pred
            else:
                pred_m = _predict_baseline_catalog(model, context["parameters"], method=method)
            m = _metrics_from_prediction(prediction=pred_m, fom_modes=fom_modes)
            bucket.append({"sample_id": sid, **{k: m.get(k) for k in m if k != "per_mode_matches"}})

    baselines = {}
    for method, rows in baseline_rows.items():
        agg = {"sample_count": len(rows)}
        for key in (
            "mic_output_proxy_p95_norm_mae",
            "radiation_proxy_p95_norm_mae",
            "radiation_proxy_rank_correlation",
            "top_k_radiation_overlap",
            "frequency_median_relative_error",
        ):
            agg.update(_aggregate_metric(rows, key))
        baselines[method] = {"aggregate": agg, "per_sample": rows}

    full_holdout_agg = {}
    for key in (
        "frequency_median_relative_error",
        "mic_output_proxy_p95_norm_mae",
        "radiation_proxy_p95_norm_mae",
        "bridge_excitation_abs_p95_norm_mae",
        "radiation_proxy_rank_correlation",
        "mic_output_proxy_rank_correlation",
        "top_k_radiation_overlap",
        "top_k_mic_overlap",
    ):
        full_holdout_agg.update(_aggregate_metric(holdout_rows, key))

    return {
        "schema": "m4_rom_intensity_v22_diagnostics_v1",
        "generated_utc": utc_now(),
        "requested_sample_ids": list(requested_sample_ids),
        "requested_sample_count": len(requested_sample_ids),
        "eligible_sample_ids": list(eligible_sample_ids),
        "eligible_completed_catalog_count": len(eligible_sample_ids),
        "eligible_sample_range": (
            f"{eligible_sample_ids[0]}..{eligible_sample_ids[-1]}"
            if eligible_sample_ids
            else None
        ),
        "k_neighbors": int(k_neighbors),
        "frequency_band_hz": list(ACCURACY_BAND_HZ),
        "learning_curve_pool_sizes": list(LEARNING_POOL_SIZES),
        "v21_full_holdout_loo": {
            "training_pool_size": max_train_available,
            "aggregate": full_holdout_agg,
            "per_sample": holdout_rows,
        },
        "learning_curve": learning_curve,
        "distance_vs_error": distance_rows,
        "repeatability_lhs_geometry": _repeatability_from_entries(eligible_entries),
        "alignment_quality": alignment_rows,
        "baselines": baselines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhs-json", type=Path, required=True)
    parser.add_argument("--samples", default="0-29")
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, default=None)
    args = parser.parse_args()

    _check_diagnostics_api()

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json.resolve())
    indices = _parse_sample_range(args.samples)
    requested_sample_ids = [_sample_id(i) for i in indices]
    eligible_sample_ids, eligible_entries, skipped = _discover_eligible_samples(
        repo_root=repo_root,
        pool=pool,
        requested_sample_ids=requested_sample_ids,
    )

    print(f"requested_sample_count={len(requested_sample_ids)}", flush=True)
    print(f"eligible_completed_catalog_count={len(eligible_sample_ids)}", flush=True)
    if eligible_sample_ids:
        print(
            f"eligible_sample_range={eligible_sample_ids[0]}..{eligible_sample_ids[-1]}",
            flush=True,
        )
    if skipped:
        print(f"skipped_requested_samples={len(skipped)}", flush=True)
        for sid in requested_sample_ids:
            if sid in skipped:
                print(f"  skip {sid}: {skipped[sid]}", flush=True)

    if not eligible_sample_ids:
        raise SystemExit("error: no eligible completed catalogs in requested sample range")

    report = run_diagnostics(
        repo_root=repo_root,
        pool=pool,
        requested_sample_ids=requested_sample_ids,
        eligible_sample_ids=eligible_sample_ids,
        eligible_entries=eligible_entries,
        k_neighbors=int(args.k_neighbors),
    )
    if skipped:
        report["skipped_requested_samples"] = skipped
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {rel(args.json_out, repo_root=repo_root)}")

    if args.csv_out is not None:
        import csv

        rows = report.get("v21_full_holdout_loo", {}).get("per_sample") or []
        if rows:
            fields = sorted({k for r in rows for k in r.keys()})
            args.csv_out.parent.mkdir(parents=True, exist_ok=True)
            with args.csv_out.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"wrote {rel(args.csv_out, repo_root=repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
