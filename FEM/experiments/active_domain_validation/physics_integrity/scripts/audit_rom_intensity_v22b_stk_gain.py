#!/usr/bin/env python3
"""ROM Intensity v2.2b — STK combined-gain diagnostic (5-sample LOO, read-only)."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
    _neighbor_catalog_from_arrays,
    build_holdout_surrogate_model,
    guitars_root,
    predict_modal_catalog,
)
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    ACCURACY_BAND_HZ,
    greedy_nearest_hz_match,
    load_fom_modes_catalog_deduped,
    resolve_sample_context,
    resolve_validation_metadata,
)
from v2_b3_m4_rom_intensity_v22 import (  # noqa: E402
    _apply_post_hoc_p95_normalization,
    _predict_frequencies_only,
    select_neighbor_mode_physics_aware,
)
from v2_b3_m4_rom_scalar_fields import (  # noqa: E402
    ACCURACY_BAND_HZ_DEFAULT,
    INTENSITY_LOG_EPSILON,
    _rank_correlation,
    _safe_float,
    blend_numeric_field,
    nearest_freq_mode_index,
    predict_mode_scalars_at_frequency,
    vote_categorical_field,
)
from v2_b3_m4_rom_stk_gain_targets import (  # noqa: E402
    COMBINED_GAIN_FIELDS,
    STRENGTH_LABELS,
    amplitude_semantics_audit,
    compute_intensity_p95_map_extended,
    enrich_catalog_stk_gains,
    enrich_mode_combined_gains,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

HOLDOUT_DEFAULT = ("sample_000", "sample_005", "sample_013", "sample_024", "sample_027")
TOP_K_FRACTIONS = (0.10, 0.20, 0.30)

METHOD_A = "A_v21_separate_idw"
METHOD_B = "B_combined_nearest_freq_idw"
METHOD_C = "C_combined_physics_aware_idw"
METHOD_D = "D_rank_percentile_idw"
METHOD_E = "E_strength_class_vote"

TARGET_SPECS: Tuple[Tuple[str, str], ...] = (
    ("mic_output_proxy_p95_norm", "mic_only"),
    ("radiation_proxy_p95_norm", "radiation_only"),
    ("bridge_excitation_abs_p95_norm", "bridge_only"),
    ("bridge_to_mic_gain_p95_norm", "bridge_x_mic"),
    ("bridge_to_radiation_gain_p95_norm", "bridge_x_radiation"),
)


def experimental_v22b_dir(repo_root: Path, shape: str) -> Path:
    return repo_root / "ROM" / shape / "experimental_v22b"


def _discover_completed(repo_root: Path, pool: Mapping[str, Any]) -> List[str]:
    ids: List[str] = []
    for entry in pool.get("entries") or []:
        sid = str(entry.get("id") or "").strip()
        if not sid:
            continue
        run_id = str(entry.get("last_run_id") or f"{sid}_{DEFAULT_RUN_ID_SUFFIX}")
        if not is_lhs_entry_completed(entry, run_id=run_id):
            continue
        cat = guitars_root(repo_root) / sid / "runs" / run_id / "aggregation" / "modes_catalog.jsonl"
        if cat.is_file():
            ids.append(sid)
    return sorted(ids)


def _kendall(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    try:
        from scipy.stats import kendalltau  # noqa: WPS433

        if len(xs) < 3:
            return None
        t, _ = kendalltau(xs, ys)
        return round(float(t), 6) if t == t else None
    except Exception:
        return None


def _topk(fom: Sequence[float], rom: Sequence[float], frac: float) -> Dict[str, Optional[float]]:
    n = len(fom)
    if n < 2:
        return {"overlap": None, "precision": None, "recall": None}
    k = max(1, int(math.ceil(n * frac)))
    fom_top = set(sorted(range(n), key=lambda i: fom[i], reverse=True)[:k])
    rom_top = set(sorted(range(n), key=lambda i: rom[i], reverse=True)[:k])
    inter = fom_top & rom_top
    return {
        "overlap": round(len(inter) / float(k), 4),
        "precision": round(len(inter) / float(len(rom_top)), 4) if rom_top else None,
        "recall": round(len(inter) / float(len(fom_top)), 4) if fom_top else None,
    }


def _ndcg(fom: Sequence[float], rom: Sequence[float], frac: float) -> Optional[float]:
    n = len(fom)
    if n < 2:
        return None
    k = max(1, int(math.ceil(n * frac)))
    rom_order = sorted(range(n), key=lambda i: rom[i], reverse=True)[:k]
    dcg = sum((2.0 ** fom[i] - 1.0) / math.log2(r + 2.0) for r, i in enumerate(rom_order))
    ideal = sorted(fom, reverse=True)[:k]
    idcg = sum((2.0 ** v - 1.0) / math.log2(r + 2.0) for r, v in enumerate(ideal))
    return round(dcg / idcg, 6) if idcg > 0 else None


def _percentile_rank(vals: Sequence[float], i: int) -> float:
    v = vals[i]
    below = sum(1 for x in vals if x < v)
    equal = sum(1 for x in vals if x == v)
    return (below + 0.5 * equal) / max(len(vals) - 1, 1)


def _band_raw_values(
    modes: Sequence[Mapping[str, Any]],
    raw_key: str,
    *,
    band: Tuple[float, float] = ACCURACY_BAND_HZ_DEFAULT,
) -> Tuple[List[float], List[int]]:
    vals: List[float] = []
    indices: List[int] = []
    for i, m in enumerate(modes):
        f_hz = _safe_float(m.get("frequency_hz"))
        v = _safe_float(m.get(raw_key))
        if f_hz is None or v is None or not (band[0] <= f_hz <= band[1]):
            continue
        vals.append(float(v))
        indices.append(i)
    return vals, indices


def _mode_percentile_rank(
    modes: Sequence[Mapping[str, Any]],
    mode_index: int,
    raw_key: str,
    *,
    band: Tuple[float, float] = ACCURACY_BAND_HZ_DEFAULT,
) -> Optional[float]:
    vals, indices = _band_raw_values(modes, raw_key, band=band)
    if not vals:
        return None
    try:
        pos = indices.index(mode_index)
    except ValueError:
        return None
    return _percentile_rank(vals, pos)


def _apply_stk_post_hoc(
    modes: List[Dict[str, Any]],
    *,
    band: Tuple[float, float] = ACCURACY_BAND_HZ_DEFAULT,
    epsilon: float = INTENSITY_LOG_EPSILON,
    percentile: int = 95,
) -> List[Dict[str, Any]]:
    for rec in modes:
        enrich_mode_combined_gains(rec)
    p95_map = compute_intensity_p95_map_extended(
        modes,
        band=band,
        percentile=percentile,
        fields_override=list(COMBINED_GAIN_FIELDS),
    )
    for rec in modes:
        for base in COMBINED_GAIN_FIELDS:
            stem = base.replace("_raw", "")
            v = _safe_float(rec.get(base))
            if v is not None:
                rec[f"{stem}_log10"] = round(math.log10(abs(v) + epsilon), 8)
            p95 = p95_map.get(base)
            if v is not None and p95 is not None and p95 > 0:
                rec[f"{stem}_p95_norm"] = round(abs(v) / p95, 8)
    return modes


def _select_mode_index(
    catalog: Sequence[Mapping[str, Any]],
    *,
    target_hz: float,
    target_class: Optional[str],
    target_region: Optional[str],
    target_shares: Tuple[Optional[float], Optional[float], Optional[float]],
    method: str,
) -> Optional[int]:
    if method in (METHOD_A, METHOD_B):
        return nearest_freq_mode_index(catalog, target_hz)
    if method == METHOD_C:
        idx, _ = select_neighbor_mode_physics_aware(
            catalog,
            target_hz=target_hz,
            target_class=target_class,
            target_region=target_region,
            target_shares=target_shares,
        )
        return idx
    return nearest_freq_mode_index(catalog, target_hz)


def _raw_field_name(target_field: str) -> str:
    if target_field == "mic_output_proxy_p95_norm":
        return "mic_output_proxy"
    if target_field == "radiation_proxy_p95_norm":
        return "radiation_proxy"
    if target_field == "bridge_excitation_abs_p95_norm":
        return "bridge_excitation_abs"
    if target_field == "bridge_to_mic_gain_p95_norm":
        return "bridge_to_mic_gain_raw"
    if target_field == "bridge_to_radiation_gain_p95_norm":
        return "bridge_to_radiation_gain_raw"
    return target_field.replace("_p95_norm", "")


def _strength_key_for(target_field: str) -> str:
    if "mic_output" in target_field:
        return "mic_output_proxy_strength_class"
    if "bridge_to_mic" in target_field:
        return "bridge_to_mic_gain_raw_strength_class"
    if "bridge_to_radiation" in target_field:
        return "bridge_to_radiation_gain_raw_strength_class"
    if "radiation" in target_field:
        return "radiation_proxy_strength_class"
    return "mic_output_proxy_strength_class"


def _neighbor_raw_value(
    catalog: Sequence[Mapping[str, Any]],
    idx: int,
    *,
    target_field: str,
    blend_method: str,
) -> Optional[float]:
    mode = catalog[idx]
    if blend_method == METHOD_A and target_field == "bridge_to_mic_gain_p95_norm":
        b = _safe_float(mode.get("bridge_excitation_abs"))
        m = _safe_float(mode.get("mic_output_proxy"))
        return float(b) * float(m) if b is not None and m is not None else None
    if blend_method == METHOD_A and target_field == "bridge_to_radiation_gain_p95_norm":
        b = _safe_float(mode.get("bridge_excitation_abs"))
        r = _safe_float(mode.get("radiation_proxy"))
        return float(b) * float(r) if b is not None and r is not None else None
    return _safe_float(mode.get(_raw_field_name(target_field)))


def _blend_field(
    *,
    target_field: str,
    target_hz: float,
    neighbor_catalogs: Sequence[Sequence[Mapping[str, Any]]],
    weights: Sequence[float],
    target_class: Optional[str],
    target_region: Optional[str],
    target_shares: Tuple[Optional[float], Optional[float], Optional[float]],
    blend_method: str,
) -> Any:
    if blend_method == METHOD_D:
        pairs: List[Tuple[float, float]] = []
        for catalog, wt in zip(neighbor_catalogs, weights):
            idx = _select_mode_index(
                catalog,
                target_hz=target_hz,
                target_class=target_class,
                target_region=target_region,
                target_shares=target_shares,
                method=METHOD_C,
            )
            if idx is None:
                continue
            vals_clean = [
                float(v)
                for j in range(len(catalog))
                if (v := _neighbor_raw_value(catalog, j, target_field=target_field, blend_method=METHOD_C))
                is not None
            ]
            if not vals_clean:
                continue
            pairs.append((_percentile_rank(vals_clean, idx), wt))
        val, _ = blend_numeric_field(pairs)
        return round(val, 8) if val is not None else None

    if blend_method == METHOD_E:
        pairs_cat: List[Tuple[str, float]] = []
        skey = _strength_key_for(target_field)
        for catalog, wt in zip(neighbor_catalogs, weights):
            idx = _select_mode_index(
                catalog,
                target_hz=target_hz,
                target_class=target_class,
                target_region=target_region,
                target_shares=target_shares,
                method=METHOD_C,
            )
            if idx is None:
                continue
            lbl = str(catalog[idx].get(skey) or "")
            if lbl:
                pairs_cat.append((lbl, wt))
        lbl, _ = vote_categorical_field(pairs_cat)
        return lbl

    match_m = METHOD_B if blend_method == METHOD_B else (METHOD_C if blend_method == METHOD_C else METHOD_A)
    pairs_num: List[Tuple[float, float]] = []
    for catalog, wt in zip(neighbor_catalogs, weights):
        idx = _select_mode_index(
            catalog,
            target_hz=target_hz,
            target_class=target_class,
            target_region=target_region,
            target_shares=target_shares,
            method=match_m,
        )
        if idx is None:
            continue
        val = _neighbor_raw_value(catalog, idx, target_field=target_field, blend_method=blend_method)
        if val is not None:
            pairs_num.append((float(val), wt))
    val, _ = blend_numeric_field(pairs_num)
    return round(val, 8) if val is not None else None


def _predict_target_catalog(
    *,
    model: Mapping[str, Any],
    parameters: Mapping[str, Any],
    method: str,
    target_field: str,
) -> Tuple[List[Dict[str, Any]], float, List[str]]:
    t0 = time.perf_counter()
    freq_out = _predict_frequencies_only(model, parameters, nev=0)
    arrays = model["arrays"]
    nn_idx = freq_out["nn_idx"]
    weights = list(freq_out["weights"])
    neighbor_ids = list(freq_out["neighbor_sample_ids"])
    neighbor_catalogs = [
        _neighbor_catalog_from_arrays(arrays, int(j), int(arrays["mode_counts"][j])) for j in nn_idx
    ]

    modes: List[Dict[str, Any]] = []
    for f_hz in freq_out["frequencies_hz"]:
        base = predict_mode_scalars_at_frequency(
            target_hz=float(f_hz),
            neighbor_catalogs=neighbor_catalogs,
            neighbor_weights=weights,
        )
        tc = base.get("coupling_class")
        tr = base.get("dominant_region")
        shares = (
            _safe_float(base.get("top_share")),
            _safe_float(base.get("back_share")),
            _safe_float(base.get("air_share")),
        )
        rec = dict(base)
        rec["frequency_hz"] = float(f_hz)
        if method == METHOD_E:
            lbl = _blend_field(
                target_field=target_field,
                target_hz=float(f_hz),
                neighbor_catalogs=neighbor_catalogs,
                weights=weights,
                target_class=str(tc) if tc else None,
                target_region=str(tr) if tr else None,
                target_shares=shares,
                blend_method=method,
            )
            rec[_strength_key_for(target_field)] = lbl
        elif method == METHOD_D:
            rec[f"{target_field}_rank_pct"] = _blend_field(
                target_field=target_field,
                target_hz=float(f_hz),
                neighbor_catalogs=neighbor_catalogs,
                weights=weights,
                target_class=str(tc) if tc else None,
                target_region=str(tr) if tr else None,
                target_shares=shares,
                blend_method=method,
            )
        else:
            raw_key = _raw_field_name(target_field)
            rec[raw_key] = _blend_field(
                target_field=target_field,
                target_hz=float(f_hz),
                neighbor_catalogs=neighbor_catalogs,
                weights=weights,
                target_class=str(tc) if tc else None,
                target_region=str(tr) if tr else None,
                target_shares=shares,
                blend_method=method,
            )
        modes.append(rec)

    if method != METHOD_E:
        modes, _ = _apply_post_hoc_p95_normalization(modes)
        modes = _apply_stk_post_hoc(modes)
    elapsed = round(time.perf_counter() - t0, 4)
    return modes, elapsed, neighbor_ids


def _evaluate(
    *,
    predicted_modes: Sequence[Mapping[str, Any]],
    fom_modes: Sequence[Mapping[str, Any]],
    target_field: str,
    method: str,
) -> Dict[str, Any]:
    rom_freqs = [float(m["frequency_hz"]) for m in predicted_modes]
    matches, _ = greedy_nearest_hz_match(rom_frequencies_hz=rom_freqs, fom_modes=fom_modes)
    fom_xs: List[float] = []
    rom_ys: List[float] = []
    log_errs: List[float] = []
    norm_errs: List[float] = []
    class_hits = 0
    class_total = 0
    strength_key = _strength_key_for(target_field)

    for m in matches:
        rom_hz = float(m["rom_frequency_hz"])
        fom_hz = float(m["fom_frequency_hz"])
        rom_mode = min(predicted_modes, key=lambda r: abs(float(r["frequency_hz"]) - rom_hz))
        fom_mode = min(fom_modes, key=lambda r: abs(float(r["frequency_hz"]) - fom_hz))
        if method == METHOD_E:
            rom_lbl = rom_mode.get(strength_key) or rom_mode.get(target_field.replace("_p95_norm", "_strength_class"))
            fom_lbl = fom_mode.get(strength_key)
            if rom_lbl and fom_lbl:
                class_total += 1
                if str(rom_lbl) == str(fom_lbl):
                    class_hits += 1
            continue
        if method == METHOD_D:
            raw_key = _raw_field_name(target_field)
            fi = min(range(len(fom_modes)), key=lambda i: abs(float(fom_modes[i]["frequency_hz"]) - fom_hz))
            fom_rank = _mode_percentile_rank(fom_modes, fi, raw_key)
            rom_rank = _safe_float(rom_mode.get(f"{target_field}_rank_pct"))
            if fom_rank is not None and rom_rank is not None:
                fom_xs.append(fom_rank)
                rom_ys.append(float(rom_rank))
            continue
        fv = _safe_float(fom_mode.get(target_field))
        rv = _safe_float(rom_mode.get(target_field))
        if fv is not None and rv is not None:
            norm_errs.append(abs(fv - rv))
        fl = _safe_float(fom_mode.get(target_field.replace("_p95_norm", "_log10")))
        rl = _safe_float(rom_mode.get(target_field.replace("_p95_norm", "_log10")))
        if fl is not None and rl is not None:
            log_errs.append(abs(fl - rl))
        raw_f = target_field.replace("_p95_norm", "_raw")
        f_raw = _safe_float(fom_mode.get(raw_f)) or _safe_float(fom_mode.get(target_field.replace("_p95_norm", "")))
        r_raw = _safe_float(rom_mode.get(raw_f)) or _safe_float(rom_mode.get(target_field.replace("_p95_norm", "")))
        if f_raw is not None and r_raw is not None:
            fom_xs.append(f_raw)
            rom_ys.append(r_raw)

    out: Dict[str, Any] = {
        "matched_mode_count": len(matches),
        "p95_norm_mae": round(statistics.mean(norm_errs), 6) if norm_errs else None,
        "log_mae": round(statistics.mean(log_errs), 6) if log_errs else None,
        "spearman": _rank_correlation(fom_xs, rom_ys),
        "kendall_tau": _kendall(fom_xs, rom_ys),
    }
    for frac in TOP_K_FRACTIONS:
        pct = int(frac * 100)
        tk = _topk(fom_xs, rom_ys, frac)
        out[f"top_{pct}pct_overlap"] = tk["overlap"]
        out[f"top_{pct}pct_precision"] = tk["precision"]
        out[f"top_{pct}pct_recall"] = tk["recall"]
        out[f"ndcg_at_{pct}pct"] = _ndcg(fom_xs, rom_ys, frac)
    if method == METHOD_E:
        out["strength_class_accuracy"] = round(class_hits / class_total, 4) if class_total else None
    return out


def _run_holdout(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    target_id: str,
    completed_ids: Sequence[str],
    k_neighbors: int,
) -> Dict[str, Any]:
    train_ids = [s for s in completed_ids if s != target_id]
    if target_id in train_ids:
        raise AssertionError(f"leakage: {target_id} in training")
    model, _training = build_holdout_surrogate_model(
        repo_root=repo_root,
        pool=pool,
        shape_name=str(pool.get("shape_name") or "classic"),
        exclude_sample_ids=[target_id],
        k_neighbors=k_neighbors,
    )
    entry = next(e for e in pool.get("entries") or [] if str(e.get("id")) == target_id)
    run_id = str(entry.get("last_run_id") or f"{target_id}_{DEFAULT_RUN_ID_SUFFIX}")
    ctx = resolve_sample_context(pool=pool, sample_id=target_id, run_id=run_id, repo_root=repo_root)
    catalog_path = Path(ctx["run_root"]) / "aggregation" / "modes_catalog.jsonl"
    _raw, deduped, _ = load_fom_modes_catalog_deduped(catalog_path)
    fom_modes, stk_meta = enrich_catalog_stk_gains(deduped, band=ACCURACY_BAND_HZ)

    vmeta = resolve_validation_metadata(
        target_sample_id=target_id,
        training_sample_ids=train_ids,
        excluded_sample_ids=[target_id],
        validation_mode="holdout",
    )

    # Frequency baseline (must match v2.1)
    freq_ref = predict_modal_catalog(model, ctx["parameters"], nev=0)
    freq_errs: List[float] = []

    results: Dict[str, Any] = {
        "sample_id": target_id,
        "validation": vmeta,
        "training_sample_count": len(train_ids),
        "fom_mode_count": len(fom_modes),
        "stk_enrichment": stk_meta,
        "methods": {},
    }

    for method in (METHOD_A, METHOD_B, METHOD_C, METHOD_D, METHOD_E):
        method_block: Dict[str, Any] = {"targets": {}, "runtime_s": 0.0, "neighbor_sample_ids": []}
        for target_field, stk_role in TARGET_SPECS:
            if method == METHOD_B and target_field in (
                "mic_output_proxy_p95_norm",
                "radiation_proxy_p95_norm",
                "bridge_excitation_abs_p95_norm",
            ):
                continue
            if method in (METHOD_B, METHOD_C) and "bridge_to" not in target_field:
                continue
            if method in (METHOD_D, METHOD_E) and stk_role not in ("bridge_x_mic", "mic_only", "bridge_x_radiation", "radiation_only"):
                continue
            pred_modes, elapsed, nids = _predict_target_catalog(
                model=model,
                parameters=ctx["parameters"],
                method=method,
                target_field=target_field,
            )
            if target_id in nids:
                raise AssertionError(f"leakage: {target_id} in neighbors for {method}")
            method_block["runtime_s"] = max(method_block["runtime_s"], elapsed)
            method_block["neighbor_sample_ids"] = nids
            if method == METHOD_A and not freq_errs:
                rom_f = [float(f) for f in (freq_ref.get("frequencies_hz") or [])]
                matches, _ = greedy_nearest_hz_match(rom_frequencies_hz=rom_f, fom_modes=fom_modes)
                freq_errs = [float(m["relative_error"]) for m in matches if m.get("relative_error") is not None]
            ev = _evaluate(
                predicted_modes=pred_modes,
                fom_modes=fom_modes,
                target_field=target_field,
                method=method,
            )
            method_block["targets"][target_field] = {"stk_role": stk_role, "metrics": ev}
        results["methods"][method] = method_block

    results["frequency_median_relative_error"] = (
        round(statistics.median(freq_errs), 6) if freq_errs else None
    )
    return results


def _aggregate(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for method in (METHOD_A, METHOD_B, METHOD_C, METHOD_D, METHOD_E):
        agg[method] = {}
        for target_field, stk_role in TARGET_SPECS:
            vals_p95 = []
            vals_spear = []
            vals_top20 = []
            for s in samples:
                m = (s.get("methods") or {}).get(method) or {}
                t = (m.get("targets") or {}).get(target_field) or {}
                met = t.get("metrics") or {}
                if met.get("p95_norm_mae") is not None:
                    vals_p95.append(float(met["p95_norm_mae"]))
                if met.get("spearman") is not None:
                    vals_spear.append(float(met["spearman"]))
                if met.get("top_20pct_overlap") is not None:
                    vals_top20.append(float(met["top_20pct_overlap"]))
            if vals_p95 or vals_spear or vals_top20:
                agg[method][target_field] = {
                    "stk_role": stk_role,
                    "p95_norm_mae_median": round(statistics.median(vals_p95), 6) if vals_p95 else None,
                    "spearman_median": round(statistics.median(vals_spear), 6) if vals_spear else None,
                    "top_20pct_overlap_median": round(statistics.median(vals_top20), 6) if vals_top20 else None,
                }
    return agg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhs-json", type=Path, default=Path("ROM/classic/lhs_pool.json"))
    parser.add_argument("--holdouts", default=",".join(HOLDOUT_DEFAULT))
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json)
    shape = str(pool.get("shape_name") or "classic")
    holdouts = [s.strip() for s in str(args.holdouts).split(",") if s.strip()]
    completed = _discover_completed(repo_root, pool)

    print(f"completed_catalog_count={len(completed)}", flush=True)
    print(f"holdout_count={len(holdouts)} holdouts={holdouts}", flush=True)

    for h in holdouts:
        if h not in completed:
            raise SystemExit(f"error: holdout {h} not in completed catalogs")

    out_dir = args.out_dir or experimental_v22b_dir(repo_root, shape)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_dir = out_dir / "per_sample"
    per_dir.mkdir(parents=True, exist_ok=True)

    samples_out: List[Dict[str, Any]] = []
    for sid in holdouts:
        print(f"[v22b] holdout {sid} ...", flush=True)
        res = _run_holdout(
            repo_root=repo_root,
            pool=pool,
            target_id=sid,
            completed_ids=completed,
            k_neighbors=int(args.k_neighbors),
        )
        samples_out.append(res)
        write_json_atomic(per_dir / f"{sid}.json", res)

    summary = {
        "schema": "m4_rom_intensity_v22b_stk_gain_v1",
        "generated_utc": utc_now(),
        "amplitude_semantics_audit": amplitude_semantics_audit(),
        "holdout_sample_ids": holdouts,
        "completed_catalog_count": len(completed),
        "validation_mode": "holdout",
        "training_includes_target": False,
        "accuracy_meaningful": True,
        "methods": [METHOD_A, METHOD_B, METHOD_C, METHOD_D, METHOD_E],
        "target_specs": [{"field": f, "stk_role": r} for f, r in TARGET_SPECS],
        "aggregate_by_method": _aggregate(samples_out),
        "per_sample_dir": rel(per_dir, repo_root=repo_root),
    }
    write_json_atomic(out_dir / "diagnostic_summary.json", summary)
    print(f"wrote {rel(out_dir / 'diagnostic_summary.json', repo_root=repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
