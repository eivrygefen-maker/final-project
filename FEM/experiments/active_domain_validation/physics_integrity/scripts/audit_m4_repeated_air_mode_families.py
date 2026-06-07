#!/usr/bin/env python3
"""Catalog + optional full-artifact audit of repeated ~281 Hz / ~390 Hz air mode families (read-only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR.parent / "docs"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_m4_lhs_pool_bridge import DEFAULT_RUN_ID_SUFFIX, lhs_entry_index, load_lhs_pool  # noqa: E402
from v2_b3_m4_lprod_interfaces import extract_geometry_dict, geometry_fingerprint  # noqa: E402
from v2_b3_m4_rom_fom_compare_lib import load_fom_modes_catalog_deduped  # noqa: E402
from v2_b3_m4_rom_scalar_fields import ROM_DEDUPE_TOLERANCE_HZ  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel  # noqa: E402

DEFAULT_LHS = "ROM/classic/lhs_pool.json"
GUITARS_REL = Path("FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars")

BAND_281 = (270.0, 290.0)
BAND_390 = (380.0, 400.0)
SPEED_OF_SOUND = 343.0
PROXY_MATCH_TOL = 1.0e-7
FREQ_ROUND_DP = 2
MIC_ROUND_DP = 10
DISCOVERY_MIN_SAMPLES = 5
DISCOVERY_BIN_HZ = 0.5

DEFAULT_CSV_OUT = DOCS_DIR / "M4_REPEATED_AIR_MODE_FAMILY_AUDIT.csv"
DEFAULT_JSON_OUT = DOCS_DIR / "M4_REPEATED_AIR_MODE_FAMILY_AUDIT.json"
DEFAULT_MD_OUT = DOCS_DIR / "M4_REPEATED_AIR_MODE_FAMILY_AUDIT.md"

FULL_ARTIFACT_DEFAULT_SAMPLES = ("sample_000", "sample_001", "sample_034", "sample_035")

CATALOG_FIELDS = (
    "sample_id",
    "family_id",
    "band",
    "frequency_hz",
    "mic_output_proxy",
    "radiation_proxy",
    "bridge_excitation_abs",
    "top_share",
    "back_share",
    "air_share",
    "coupling_class",
    "dominant_region",
    "mic_output_method",
    "chunk_id",
    "target_hz",
    "chunk_interval_hz",
    "raw_occurrence_count",
    "deduped_occurrence_count",
    "dedupe_merge_count",
    "provenance_chunk_ids",
    "origin",
)


def _parse_sample_ids(arg: str) -> List[str]:
    out: List[str] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and part.replace("-", "").isdigit():
            lo, hi = part.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                out.append(f"sample_{i:03d}")
        elif part.isdigit():
            out.append(f"sample_{int(part):03d}")
        else:
            out.append(part)
    return out


def _sha256_file(path: Path, *, max_bytes: Optional[int] = None) -> Optional[str]:
    if not path.is_file():
        return None
    data = path.read_bytes() if max_bytes is None else path.read_bytes()[:max_bytes]
    return hashlib.sha256(data).hexdigest()


def _sha256_indices(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(i)) for i in sorted(indices))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _estimate_cavity_volume_m3(geom: Mapping[str, float]) -> Optional[float]:
    """Box cavity proxy: L×W×(D−2×top_thickness). Not CAD-exact."""
    try:
        length = float(geom["length"])
        width = float(geom["width"])
        depth = float(geom["depth"])
        top_t = float(geom.get("top_thickness") or 0.003)
        inner = max(depth - 2.0 * top_t, 1.0e-6)
        return length * width * inner
    except (KeyError, TypeError, ValueError):
        return None


def _hole_area_m2(geom: Mapping[str, float]) -> Optional[float]:
    r = _safe_float(geom.get("hole_radius"))
    if r is None or r <= 0:
        return None
    return math.pi * r * r


def _helmholtz_estimate_hz(geom: Mapping[str, float]) -> Optional[float]:
    """
    Helmholtz-style diagnostic (not CAD-exact):
      f_H = (c / 2π) * sqrt(A / (V * L_eff))
    Approximations:
      A = π * hole_radius^2
      V = L * W * (D - 2*top_thickness)
      L_eff = top_thickness + 0.01 m  (plate + short neck proxy)
    """
    area = _hole_area_m2(geom)
    volume = _estimate_cavity_volume_m3(geom)
    top_t = _safe_float(geom.get("top_thickness")) or 0.003
    if area is None or volume is None or volume <= 0:
        return None
    l_eff = max(top_t + 0.01, 1.0e-4)
    return (SPEED_OF_SOUND / (2.0 * math.pi)) * math.sqrt(area / (volume * l_eff))


def _scale_indicators_hz(geom: Mapping[str, float]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for key, dim in (("c_over_2L", "length"), ("c_over_2W", "width"), ("c_over_2D", "depth")):
        d = _safe_float(geom.get(dim))
        out[key] = (SPEED_OF_SOUND / (2.0 * d)) if d and d > 0 else None
    return out


def _geom_features(entry: Mapping[str, Any]) -> Dict[str, Any]:
    geom = extract_geometry_dict(entry)
    vol = _estimate_cavity_volume_m3(geom)
    hole_a = _hole_area_m2(geom)
    params = entry.get("parameters") or {}
    return {
        "geometry": geom,
        "geometry_fingerprint": geometry_fingerprint(geom) if geom else None,
        "length": geom.get("length"),
        "width": geom.get("width"),
        "depth": geom.get("depth"),
        "hole_radius": geom.get("hole_radius"),
        "top_thickness": geom.get("top_thickness"),
        "back_thickness": geom.get("back_thickness"),
        "top_wood_id": params.get("top_wood_id"),
        "back_wood_id": params.get("back_wood_id"),
        "hole_area_m2": hole_a,
        "est_cavity_volume_m3": vol,
        "hole_area_over_volume": (hole_a / vol) if hole_a and vol and vol > 0 else None,
        "length_over_width": (
            float(geom["length"]) / float(geom["width"])
            if geom.get("length") and geom.get("width") and float(geom["width"]) > 0
            else None
        ),
        "helmholtz_estimate_hz": _helmholtz_estimate_hz(geom),
        "scale_indicators_hz": _scale_indicators_hz(geom),
    }


def _run_root(repo_root: Path, sample_id: str, run_id: str) -> Path:
    return repo_root / GUITARS_REL / sample_id / "runs" / run_id


def _resolve_run_id(pool: Mapping[str, Any], sample_id: str, run_id_suffix: str) -> Tuple[str, Dict[str, Any]]:
    idx = lhs_entry_index(pool, sample_id)
    entry = (pool.get("entries") or [])[idx] if idx is not None else {}
    run_id = str(entry.get("last_run_id") or f"{sample_id}_{run_id_suffix}")
    return run_id, entry


def _mode_in_band(freq: float, band: Tuple[float, float]) -> bool:
    return band[0] <= freq <= band[1]


def _count_raw_near(
    raw_modes: Sequence[Mapping[str, Any]],
    freq_hz: float,
    *,
    tol_hz: float = ROM_DEDUPE_TOLERANCE_HZ,
) -> int:
    return sum(1 for m in raw_modes if abs(float(m["frequency_hz"]) - freq_hz) <= tol_hz)


def _chunk_interval_from_mode(m: Mapping[str, Any]) -> Optional[List[float]]:
    win = m.get("window_hz") or m.get("chunk_interval_hz")
    if isinstance(win, (list, tuple)) and len(win) == 2:
        return [float(win[0]), float(win[1])]
    return None


def _extract_mode_row(
    *,
    sample_id: str,
    family_id: str,
    band: str,
    mode: Mapping[str, Any],
    raw_modes: Sequence[Mapping[str, Any]],
    origin: str,
) -> Dict[str, Any]:
    freq = float(mode["frequency_hz"])
    prov_chunks = mode.get("provenance_chunk_ids")
    if not prov_chunks and mode.get("chunk_id"):
        prov_chunks = [str(mode["chunk_id"])]
    return {
        "sample_id": sample_id,
        "family_id": family_id,
        "band": band,
        "frequency_hz": freq,
        "mic_output_proxy": mode.get("mic_output_proxy"),
        "radiation_proxy": mode.get("radiation_proxy"),
        "bridge_excitation_abs": mode.get("bridge_excitation_abs"),
        "top_share": mode.get("top_share"),
        "back_share": mode.get("back_share"),
        "air_share": mode.get("air_share"),
        "coupling_class": mode.get("coupling_class"),
        "dominant_region": mode.get("dominant_region"),
        "mic_output_method": mode.get("mic_output_method"),
        "chunk_id": mode.get("chunk_id"),
        "target_hz": mode.get("target_hz"),
        "chunk_interval_hz": _chunk_interval_from_mode(mode),
        "raw_occurrence_count": _count_raw_near(raw_modes, freq),
        "deduped_occurrence_count": 1 if origin == "deduped_peak" else None,
        "dedupe_merge_count": mode.get("provenance_count") or 1,
        "provenance_chunk_ids": prov_chunks,
        "origin": origin,
    }


def _band_cluster_modes(
    modes: Sequence[Mapping[str, Any]],
    *,
    band: Tuple[float, float],
    prefer_air: bool = True,
    n: int = 8,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for m in modes:
        f = _safe_float(m.get("frequency_hz"))
        mic = _safe_float(m.get("mic_output_proxy"))
        if f is None or mic is None or not _mode_in_band(f, band):
            continue
        air = _safe_float(m.get("air_share")) or 0.0
        if prefer_air and air < 0.5:
            continue
        rows.append(dict(m))
    rows.sort(key=lambda r: (-float(r.get("mic_output_proxy") or 0.0), float(r["frequency_hz"])))
    return rows[:n]


def _discover_repeated_families(
    per_sample_peaks: Sequence[Mapping[str, Any]],
    *,
    min_samples: int = DISCOVERY_MIN_SAMPLES,
    bin_hz: float = DISCOVERY_BIN_HZ,
) -> List[Dict[str, Any]]:
    """Bin strongest per-sample peaks; return bins repeated across many samples."""
    bins: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_sample_peaks:
        f = _safe_float(row.get("peak_freq_hz"))
        if f is None:
            continue
        b = round(f / bin_hz) * bin_hz
        bins[b].append(row)

    out: List[Dict[str, Any]] = []
    for center, items in sorted(bins.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(items) < min_samples:
            continue
        freqs = [float(x["peak_freq_hz"]) for x in items]
        mics = [float(x["peak_mic"]) for x in items if x.get("peak_mic") is not None]
        out.append(
            {
                "bin_center_hz": center,
                "sample_count": len(items),
                "freq_mean_hz": mean(freqs),
                "freq_std_hz": pstdev(freqs) if len(freqs) > 1 else 0.0,
                "freq_range_hz": max(freqs) - min(freqs),
                "mic_mean": mean(mics) if mics else None,
                "mic_std": pstdev(mics) if len(mics) > 1 else 0.0,
                "mic_range": (max(mics) - min(mics)) if mics else None,
                "exact_mic_values_10dp": len({round(m, 10) for m in mics}) if mics else 0,
                "samples": [x["sample_id"] for x in items],
            }
        )
    return out


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return None
    return num / (den_x * den_y)


def _linregress(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, Optional[float]]:
    if len(xs) < 3 or len(xs) != len(ys):
        return {"slope": None, "intercept": None, "r": None}
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return {"slope": None, "intercept": None, "r": _pearson(xs, ys)}
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    return {"slope": slope, "intercept": intercept, "r": _pearson(xs, ys)}


def _family_stats(rows: Sequence[Mapping[str, Any]], *, key_freq: str = "frequency_hz") -> Dict[str, Any]:
    freqs = [_safe_float(r.get(key_freq)) for r in rows]
    mics = [_safe_float(r.get("mic_output_proxy")) for r in rows]
    freqs = [f for f in freqs if f is not None]
    mics = [m for m in mics if m is not None]
    if not freqs:
        return {}
    exact_mic = len({round(m, MIC_ROUND_DP) for m in mics}) == 1 if mics else False
    near_mic = len({round(m, 6) for m in mics}) == 1 if mics else False
    return {
        "sample_count": len(rows),
        "frequency_mean_hz": mean(freqs),
        "frequency_std_hz": pstdev(freqs) if len(freqs) > 1 else 0.0,
        "frequency_range_hz": max(freqs) - min(freqs),
        "mic_mean": mean(mics) if mics else None,
        "mic_std": pstdev(mics) if len(mics) > 1 else 0.0,
        "mic_range": (max(mics) - min(mics)) if mics else None,
        "exact_mic_repeats_10dp": exact_mic,
        "near_mic_repeats_6dp": near_mic,
    }


def _duplicate_exact_groups(raw_modes: Sequence[Mapping[str, Any]], band: Tuple[float, float]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[float, float], List[Dict[str, Any]]] = defaultdict(list)
    for m in raw_modes:
        f = _safe_float(m.get("frequency_hz"))
        mic = _safe_float(m.get("mic_output_proxy"))
        if f is None or mic is None or not _mode_in_band(f, band):
            continue
        key = (round(f, 6), round(mic, MIC_ROUND_DP))
        groups[key].append(dict(m))
    out = []
    for (f, mic), items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0][0])):
        if len(items) < 2:
            continue
        out.append(
            {
                "frequency_hz": f,
                "mic_output_proxy": mic,
                "duplicate_count": len(items),
                "chunk_ids": sorted({str(x.get("chunk_id") or "") for x in items if x.get("chunk_id")}),
            }
        )
    return out


def _load_target_plan(run_root: Path) -> Dict[str, Any]:
    for name in ("lprod_target_plan.json", "lprod_target_plan.placeholder.json"):
        path = run_root / "lprod" / name
        if path.is_file():
            try:
                return load_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
    return {}


def _targets_near(plan: Mapping[str, Any], center_hz: float, *, tol: float = 8.0) -> List[float]:
    out: List[float] = []
    for t in plan.get("targets_hz") or []:
        hz = _safe_float(t)
        if hz is not None and abs(hz - center_hz) <= tol:
            out.append(hz)
    return sorted(out)


def _chunk_target_audit_for_sample(
    *,
    sample_id: str,
    run_root: Path,
    raw_modes: Sequence[Mapping[str, Any]],
    bands: Sequence[Tuple[str, Tuple[float, float]]],
) -> List[Dict[str, Any]]:
    plan = _load_target_plan(run_root)
    plan_targets = [_safe_float(t) for t in (plan.get("targets_hz") or [])]
    plan_targets = [t for t in plan_targets if t is not None]

    chunk_plan_path = run_root / "lprod" / "worker_chunk_plan.preview.json"
    chunk_ranges: Dict[str, List[float]] = {}
    if chunk_plan_path.is_file():
        try:
            cp = load_json(chunk_plan_path)
            for ch in cp.get("chunks") or []:
                cid = str(ch.get("chunk_id") or "")
                fr = ch.get("freq_range_hz")
                if cid and isinstance(fr, (list, tuple)) and len(fr) == 2:
                    chunk_ranges[cid] = [float(fr[0]), float(fr[1])]
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    rows: List[Dict[str, Any]] = []
    for band_name, band in bands:
        for m in raw_modes:
            f = _safe_float(m.get("frequency_hz"))
            if f is None or not _mode_in_band(f, band):
                continue
            target = _safe_float(m.get("target_hz"))
            chunk_id = str(m.get("chunk_id") or "")
            dist = abs(f - target) if target is not None else None
            nearest_plan = None
            if plan_targets:
                nearest_plan = min(plan_targets, key=lambda t: abs(t - f))
            rows.append(
                {
                    "sample": sample_id,
                    "band": band_name,
                    "mode_frequency_hz": f,
                    "source_chunk": chunk_id,
                    "chunk_freq_range_hz": chunk_ranges.get(chunk_id),
                    "target_shift_hz": target,
                    "nearest_plan_target_hz": nearest_plan,
                    "distance_from_target_hz": dist,
                    "distance_from_nearest_plan_hz": (
                        abs(f - nearest_plan) if nearest_plan is not None else None
                    ),
                    "mic_output_proxy": m.get("mic_output_proxy"),
                    "air_share": m.get("air_share"),
                    "raw_occurrence_count": 1,
                }
            )

    # Collapse identical (sample, freq, chunk, target) keys
    collapsed: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for r in rows:
        key = (
            r["sample"],
            round(float(r["mode_frequency_hz"]), 6),
            r["source_chunk"],
            r["target_shift_hz"],
        )
        if key not in collapsed:
            collapsed[key] = dict(r)
        else:
            collapsed[key]["raw_occurrence_count"] = int(collapsed[key]["raw_occurrence_count"]) + 1
    return sorted(collapsed.values(), key=lambda r: (r["sample"], r["mode_frequency_hz"]))


def _deduped_merge_near(
    deduped: Sequence[Mapping[str, Any]],
    freq_hz: float,
    *,
    tol_hz: float = ROM_DEDUPE_TOLERANCE_HZ,
) -> Optional[Dict[str, Any]]:
    best = None
    best_d = float("inf")
    for m in deduped:
        f = _safe_float(m.get("frequency_hz"))
        if f is None:
            continue
        d = abs(f - freq_hz)
        if d <= tol_hz and d < best_d:
            best_d = d
            best = m
    return best


def audit_sample_catalog(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id_suffix: str,
) -> Dict[str, Any]:
    run_id, entry = _resolve_run_id(pool, sample_id, run_id_suffix)
    run_root = _run_root(repo_root, sample_id, run_id)
    catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
    geom_feat = _geom_features(entry)

    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "run_id": run_id,
        "catalog_path": rel(catalog_path, repo_root=repo_root) if catalog_path.is_file() else None,
        "catalog_status": "missing",
        **{k: v for k, v in geom_feat.items() if k != "geometry"},
    }

    if not catalog_path.is_file():
        return row

    raw_modes, deduped, dedupe_meta = load_fom_modes_catalog_deduped(catalog_path)
    row.update(
        {
            "catalog_status": "ok",
            "raw_mode_count": len(raw_modes),
            "deduped_mode_count": len(deduped),
            "dedupe_merge_groups": dedupe_meta.get("dedupe_merge_groups"),
            "dedupe_tolerance_hz": ROM_DEDUPE_TOLERANCE_HZ,
        }
    )

    cluster_rows: List[Dict[str, Any]] = []
    for band_name, band in (("281", BAND_281), ("390", BAND_390)):
        raw_peaks = _band_cluster_modes(raw_modes, band=band, n=6)
        dedup_peaks = _band_cluster_modes(deduped, band=band, n=3)
        family_id = f"band_{band_name}"
        for i, m in enumerate(raw_peaks):
            cluster_rows.append(
                _extract_mode_row(
                    sample_id=sample_id,
                    family_id=family_id,
                    band=band_name,
                    mode=m,
                    raw_modes=raw_modes,
                    origin=f"raw_rank_{i + 1}",
                )
            )
        if dedup_peaks:
            cluster_rows.append(
                _extract_mode_row(
                    sample_id=sample_id,
                    family_id=family_id,
                    band=band_name,
                    mode=dedup_peaks[0],
                    raw_modes=raw_modes,
                    origin="deduped_peak",
                )
            )

        dups = _duplicate_exact_groups(raw_modes, band)
        row[f"band_{band_name}_exact_duplicate_groups"] = len(dups)
        row[f"band_{band_name}_largest_duplicate_group"] = dups[0]["duplicate_count"] if dups else 0

        top_raw = raw_peaks[0] if raw_peaks else None
        if top_raw:
            row[f"peak_{band_name}_freq_hz"] = top_raw.get("frequency_hz")
            row[f"peak_{band_name}_mic"] = top_raw.get("mic_output_proxy")
            row[f"peak_{band_name}_air_share"] = top_raw.get("air_share")
            row[f"peak_{band_name}_coupling_class"] = top_raw.get("coupling_class")
            row[f"peak_{band_name}_chunk_id"] = top_raw.get("chunk_id")
            row[f"peak_{band_name}_target_hz"] = top_raw.get("target_hz")

    row["cluster_modes"] = cluster_rows

    plan = _load_target_plan(run_root)
    row["lprod_targets_near_281_hz"] = _targets_near(plan, 281.0)
    row["lprod_targets_near_390_hz"] = _targets_near(plan, 390.0)
    row["lprod_target_count"] = len(plan.get("targets_hz") or [])

    agg_path = run_root / "aggregation" / "aggregation_result.json"
    if agg_path.is_file():
        try:
            agg = load_json(agg_path)
            row["agg_raw_mode_count"] = agg.get("raw_mode_count")
            row["agg_deduped_mode_count"] = agg.get("deduped_mode_count")
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    return row


def _audit_soundhole_mask(run_root: Path) -> Dict[str, Any]:
    """Inspect region_dof_indices.npz for mic proxy DOF selection (read-only)."""
    out: Dict[str, Any] = {"mask_method": "soundhole_displacement_rms_proxy_v1 (u_idx_soundhole)"}
    ckpt = run_root / "lprod" / "checkpoint"
    npz_path = ckpt / "region_dof_indices.npz"
    built_path = ckpt / "built_metadata.json"
    out["region_dof_indices_present"] = npz_path.is_file()
    out["built_metadata_present"] = built_path.is_file()
    if not npz_path.is_file():
        out["mask_status"] = "region_dof_indices_missing"
        if built_path.is_file():
            try:
                built = load_json(built_path)
                p_idx = built.get("p_idx") or []
                out["fallback_p_idx_air_count"] = len(p_idx)
                out["note"] = "Without npz, mic proxy may fall back to cavity pressure if structural DOFs unavailable."
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return out

    try:
        import numpy as np  # noqa: WPS433

        with np.load(npz_path, allow_pickle=False) as z:
            sh = np.asarray(z.get("u_idx_soundhole", []), dtype=np.int32).ravel()
            out["n_soundhole_dofs"] = int(sh.size)
            out["soundhole_index_sha256"] = _sha256_indices(sh.tolist())
            out["soundhole_index_min"] = int(sh.min()) if sh.size else None
            out["soundhole_index_max"] = int(sh.max()) if sh.size else None
            for k in ("u_idx_top", "u_idx_back", "p_idx_air"):
                arr = np.asarray(z.get(k, []), dtype=np.int32).ravel()
                out[f"n_{k}"] = int(arr.size)
            out["region_dof_source"] = (
                str(np.asarray(z["region_dof_source"]).item()) if "region_dof_source" in z.files else None
            )
    except Exception as exc:  # noqa: BLE001
        out["mask_status"] = f"load_error:{type(exc).__name__}"
        return out

    out["mask_status"] = "ok"
    return out


def audit_full_artifacts(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id_suffix: str,
) -> Dict[str, Any]:
    run_id, entry = _resolve_run_id(pool, sample_id, run_id_suffix)
    run_root = _run_root(repo_root, sample_id, run_id)
    geom_feat = _geom_features(entry)

    out: Dict[str, Any] = {
        "sample_id": sample_id,
        "run_id": run_id,
        "run_root": rel(run_root, repo_root=repo_root),
        **{k: v for k, v in geom_feat.items() if k != "geometry"},
        "geometry_dict": geom_feat.get("geometry"),
    }

    sample_input = run_root / "sample" / "sample_input.json"
    if sample_input.is_file():
        try:
            out["sample_input_geometry"] = extract_geometry_dict(load_json(sample_input))
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    mesh_dir = run_root / "lprod" / "mesh" / "L_prod"
    mesh_summary_path = mesh_dir / f"{sample_id}_mesh_build_summary.json"
    if mesh_summary_path.is_file():
        try:
            summary = load_json(mesh_summary_path)
            out["mesh_build_summary"] = {
                "n_nodes": summary.get("n_nodes"),
                "n_tetrahedra": summary.get("n_tetrahedra"),
                "geometry": summary.get("geometry"),
                "mesh_path": summary.get("mesh_path"),
            }
            mpath = Path(str(summary.get("mesh_path") or ""))
            if not mpath.is_file():
                mpath = mesh_dir / f"{sample_id}.msh"
            out["mesh_path"] = str(mpath)
            out["mesh_sha256"] = _sha256_file(mpath, max_bytes=2_000_000)
            out["mesh_size_bytes"] = mpath.stat().st_size if mpath.is_file() else None
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    else:
        out["mesh_build_summary"] = None
        out["mesh_status"] = "missing_or_compacted"

    ckpt = run_root / "lprod" / "checkpoint"
    built_path = ckpt / "built_metadata.json"
    if built_path.is_file():
        try:
            built = load_json(built_path)
            out["checkpoint"] = {
                "built_metadata_sha256": _sha256_file(built_path),
                "n_u_b3": built.get("n_u_b3"),
                "n_w": built.get("n_w"),
                "active_dimension": built.get("active_dimension"),
                "n_p_air_estimate": len(built.get("p_idx") or []),
                "mesh_level": built.get("mesh_level"),
                "region_dof_mesh_file": built.get("region_dof_mesh_file"),
            }
            for mat in ("A_active_csr.npz", "M_active_csr.npz", "A.npz", "M.npz"):
                mp = ckpt / mat
                if mp.is_file():
                    out["checkpoint"][f"{mat}_sha256_sample"] = _sha256_file(mp, max_bytes=1_000_000)
                    out["checkpoint"][f"{mat}_bytes"] = mp.stat().st_size
        except (OSError, ValueError, json.JSONDecodeError):
            out["checkpoint"] = {"status": "built_metadata_unreadable"}
    else:
        out["checkpoint"] = {"status": "missing_or_compacted"}

    out["soundhole_mic_mask"] = _audit_soundhole_mask(run_root)

    catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
    if catalog_path.is_file():
        raw_modes, deduped, _ = load_fom_modes_catalog_deduped(catalog_path)
        for band_name, band in (("281", BAND_281), ("390", BAND_390)):
            peak = _band_cluster_modes(deduped or raw_modes, band=band, n=1)
            if peak:
                m = peak[0]
                out[f"mode_localization_{band_name}"] = {
                    "frequency_hz": m.get("frequency_hz"),
                    "air_share": m.get("air_share"),
                    "top_share": m.get("top_share"),
                    "back_share": m.get("back_share"),
                    "dominant_region": m.get("dominant_region"),
                    "coupling_class": m.get("coupling_class"),
                    "mic_output_proxy": m.get("mic_output_proxy"),
                    "mic_output_method": m.get("mic_output_method"),
                    "radiation_proxy": m.get("radiation_proxy"),
                    "participation_status": m.get("participation_status"),
                }

    return out


def _correlation_block(rows: Sequence[Mapping[str, Any]], *, band_key: str) -> Dict[str, Any]:
    pts: List[Dict[str, float]] = []
    for r in rows:
        if r.get("catalog_status") != "ok":
            continue
        f = _safe_float(r.get(f"peak_{band_key}_freq_hz"))
        if f is None:
            continue
        pts.append(
            {
                "freq": f,
                "mic": _safe_float(r.get(f"peak_{band_key}_mic")) or 0.0,
                "helm": _safe_float(r.get("helmholtz_estimate_hz")) or 0.0,
                "volume": _safe_float(r.get("est_cavity_volume_m3")) or 0.0,
                "hole_av": _safe_float(r.get("hole_area_over_volume")) or 0.0,
                "length": _safe_float(r.get("length")) or 0.0,
                "width": _safe_float(r.get("width")) or 0.0,
                "depth": _safe_float(r.get("depth")) or 0.0,
                "hole_radius": _safe_float(r.get("hole_radius")) or 0.0,
            }
        )
    if len(pts) < 3:
        return {"n": len(pts)}

    def _series(key: str) -> List[float]:
        return [p[key] for p in pts]

    freq = _series("freq")
    return {
        "n": len(pts),
        "freq_vs_helmholtz": _linregress(_series("helm"), freq),
        "freq_vs_volume": _linregress(_series("volume"), freq),
        "freq_vs_hole_area_over_volume": _linregress(_series("hole_av"), freq),
        "freq_vs_length": _linregress(_series("length"), freq),
        "freq_vs_hole_radius": _linregress(_series("hole_radius"), freq),
        "mic_vs_volume": _linregress(_series("volume"), _series("mic")),
        "mic_vs_hole_radius": _linregress(_series("hole_radius"), _series("mic")),
    }


def _compare_sample_000(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    s0 = next((r for r in rows if r.get("sample_id") == "sample_000"), None)
    others = [r for r in rows if r.get("sample_id") != "sample_000" and r.get("catalog_status") == "ok"]
    if not s0 or s0.get("catalog_status") != "ok":
        return {"status": "sample_000_missing_or_incomplete"}

    def _dist(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return abs(a - b)

    out: Dict[str, Any] = {"sample_000": {}, "vs_others_median": {}}
    for band in ("281", "390"):
        f0 = _safe_float(s0.get(f"peak_{band}_freq_hz"))
        m0 = _safe_float(s0.get(f"peak_{band}_mic"))
        freqs = [_safe_float(r.get(f"peak_{band}_freq_hz")) for r in others]
        mics = [_safe_float(r.get(f"peak_{band}_mic")) for r in others]
        freqs = [f for f in freqs if f is not None]
        mics = [m for m in mics if m is not None]
        out["sample_000"][band] = {"freq_hz": f0, "mic": m0, "air_share": s0.get(f"peak_{band}_air_share")}
        if freqs:
            med_f = sorted(freqs)[len(freqs) // 2]
            med_m = sorted(mics)[len(mics) // 2] if mics else None
            out["vs_others_median"][band] = {
                "median_freq_hz": med_f,
                "median_mic": med_m,
                "freq_delta_from_median_hz": _dist(f0, med_f),
                "mic_delta_from_median": _dist(m0, med_m),
                "geometry_fingerprint_differs": s0.get("geometry_fingerprint")
                not in {r.get("geometry_fingerprint") for r in others},
            }
    return out


def _geometry_reuse_flags(full_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    mesh_hashes: Dict[str, List[str]] = defaultdict(list)
    ckpt_hashes: Dict[str, List[str]] = defaultdict(list)
    sh_hashes: Dict[str, List[str]] = defaultdict(list)
    fps: Dict[str, List[str]] = defaultdict(list)

    for r in full_rows:
        sid = str(r.get("sample_id"))
        fp = r.get("geometry_fingerprint")
        if fp:
            fps[str(fp)].append(sid)
        mh = (r.get("mesh_sha256") or "")
        if mh:
            mesh_hashes[str(mh)].append(sid)
        ck = ((r.get("checkpoint") or {}).get("built_metadata_sha256") or "")
        if ck:
            ckpt_hashes[str(ck)].append(sid)
        sh = ((r.get("soundhole_mic_mask") or {}).get("soundhole_index_sha256") or "")
        if sh:
            sh_hashes[str(sh)].append(sid)

    def _dup_map(d: Mapping[str, List[str]]) -> Dict[str, List[str]]:
        return {k: v for k, v in d.items() if len(v) > 1}

    return {
        "duplicate_geometry_fingerprint_groups": _dup_map(fps),
        "duplicate_mesh_sha256_groups": _dup_map(mesh_hashes),
        "duplicate_checkpoint_metadata_sha256_groups": _dup_map(ckpt_hashes),
        "duplicate_soundhole_index_sha256_groups": _dup_map(sh_hashes),
    }


def _target_coincidence_summary(
    chunk_rows: Sequence[Mapping[str, Any]],
    *,
    centers: Sequence[float] = (281.0, 390.0),
    tol: float = 0.1,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for center in centers:
        near = [
            r
            for r in chunk_rows
            if _safe_float(r.get("target_shift_hz")) is not None
            and abs(float(r["target_shift_hz"]) - center) <= tol
        ]
        out[str(int(center))] = {
            "modes_with_target_within_0p1_hz": len(near),
            "fraction_of_chunk_rows": (len(near) / len(chunk_rows)) if chunk_rows else 0.0,
            "distinct_target_values": sorted(
                {round(float(r["target_shift_hz"]), 4) for r in near if r.get("target_shift_hz") is not None}
            ),
        }
    return out


def _determine_verdict(report: Mapping[str, Any]) -> str:
    """Single primary classification from aggregated signals."""
    cross = report.get("cross_sample") or {}
    c281 = cross.get("cluster_281hz") or {}
    c390 = cross.get("cluster_390hz") or {}
    reuse = report.get("full_artifact_reuse") or {}
    target_hit = report.get("target_coincidence") or {}
    corr281 = (report.get("correlations") or {}).get("band_281") or {}

    mesh_dup = reuse.get("duplicate_mesh_sha256_groups") or {}
    ckpt_dup = reuse.get("duplicate_checkpoint_metadata_sha256_groups") or {}
    if mesh_dup or ckpt_dup:
        return "GEOMETRY_PROPAGATION_SUSPECTED"

    t281 = target_hit.get("281") or {}
    if t281.get("modes_with_target_within_0p1_hz", 0) >= 10:
        return "PLOT_OR_DEDUPE_ARTIFACT"

    freq_span = _safe_float(c281.get("frequency_range_hz")) or 999.0
    mic_exact = bool(c281.get("exact_mic_repeats_10dp"))
    mic_rel = _safe_float(c281.get("mic_rel_span"))
    helm_r = _safe_float((corr281.get("freq_vs_helmholtz") or {}).get("r"))

    if mic_exact and freq_span < 0.05:
        return "PROXY_INSENSITIVITY_SUSPECTED"

    dup_groups = _safe_float(cross.get("total_exact_duplicate_groups_281")) or 0.0
    if dup_groups >= 5:
        return "PLOT_OR_DEDUPE_ARTIFACT"

    if helm_r is not None and abs(helm_r) >= 0.45 and freq_span > 0.5:
        return "PHYSICALLY_PLAUSIBLE"

    if freq_span < 0.02 and (c281.get("sample_count") or 0) >= 10:
        if mic_rel is not None and mic_rel < 0.01:
            return "PROXY_INSENSITIVITY_SUSPECTED"
        return "FIXED_AIR_DOMAIN_SUSPECTED"

    if (c281.get("sample_count") or 0) >= 5 and (c390.get("sample_count") or 0) >= 5:
        return "PHYSICALLY_PLAUSIBLE"

    return "INCONCLUSIVE"


def _render_markdown(report: Mapping[str, Any]) -> str:
    verdict = report.get("verdict") or "INCONCLUSIVE"
    lines = [
        "# M4 repeated air mode family audit",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "Read-only audit of ~281 Hz and ~390 Hz `mic_output_proxy` peak families across completed samples.",
        "No FOM, ROM, physics, solver, or production data were modified.",
        "",
        "## Executive summary",
        "",
    ]

    cross = report.get("cross_sample") or {}
    for label, key in (("~281 Hz", "cluster_281hz"), ("~390 Hz", "cluster_390hz")):
        c = cross.get(key) or {}
        if not c:
            continue
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Samples with peak: **{c.get('sample_count')}**",
                f"- Frequency mean/std/range: **{c.get('frequency_mean_hz'):.4f}** / "
                f"**{c.get('frequency_std_hz'):.6f}** / **{c.get('frequency_range_hz'):.6f}** Hz",
                f"- Mic proxy mean/std/range: **{c.get('mic_mean'):.6e}** / "
                f"**{c.get('mic_std'):.6e}** / **{c.get('mic_range'):.6e}**",
                f"- Exact mic repeats (10 dp): **{c.get('exact_mic_repeats_10dp')}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Distinction (validity layers)",
            "",
            "| Layer | This audit scope |",
            "|-------|------------------|",
            "| Frequency validity | Eigenvalue locations vs targets/chunks |",
            "| Mode classification | `coupling_class`, `dominant_region`, shares |",
            "| Mic proxy validity | `mic_output_proxy` / soundhole mask sensitivity |",
            "| Duplicate/plot validity | Raw vs deduped catalog, chunk provenance |",
            "| ROM training impact | Deduped catalog used by ROM compare — separate regen if dedupe-only fix |",
            "",
            "## Task 1 — Catalog audit",
            "",
            f"- Samples audited: `{', '.join(report.get('sample_ids') or [])}`",
            f"- Dedupe tolerance: **{report.get('dedupe_tolerance_hz')} Hz** (production semantics via `load_fom_modes_catalog_deduped`)",
            "",
            "### Data-driven repeated families",
            "",
        ]
    )
    for fam in (report.get("discovered_families") or [])[:8]:
        lines.append(
            f"- bin **{fam.get('bin_center_hz')} Hz**: {fam.get('sample_count')} samples, "
            f"freq range **{fam.get('freq_range_hz'):.4f} Hz**, "
            f"mic range **{fam.get('mic_range'):.6e}**"
        )
    lines.append("")

    s000 = report.get("sample_000_comparison") or {}
    if s000.get("sample_000"):
        lines.extend(["### sample_000 vs group", "", "```json", json.dumps(s000, indent=2), "```", ""])

    corr = report.get("correlations") or {}
    if corr:
        lines.extend(["### Correlations / regressions", "", "```json", json.dumps(corr, indent=2), "```", ""])

    lines.extend(
        [
            "## Task 2 — Full-artifact audit (retained heavy samples)",
            "",
        ]
    )
    for fr in report.get("full_artifact_rows") or []:
        lines.append(f"### {fr.get('sample_id')}")
        lines.append(f"- geometry fingerprint: `{fr.get('geometry_fingerprint')}`")
        lines.append(f"- mesh_sha256 (sampled): `{fr.get('mesh_sha256')}`")
        ck = fr.get("checkpoint") or {}
        lines.append(f"- checkpoint: `{ck.get('status') or 'present'}`")
        mask = fr.get("soundhole_mic_mask") or {}
        lines.append(
            f"- soundhole mask DOFs: **{mask.get('n_soundhole_dofs')}**, "
            f"index hash: `{mask.get('soundhole_index_sha256')}`"
        )
        lines.append("")

    reuse = report.get("full_artifact_reuse") or {}
    if reuse:
        lines.extend(["### Cross-sample reuse flags", "", "```json", json.dumps(reuse, indent=2), "```", ""])

    lines.extend(
        [
            "## Task 3 — Solver target/chunk audit",
            "",
        ]
    )
    tc = report.get("target_coincidence") or {}
    if tc:
        lines.extend(["```json", json.dumps(tc, indent=2), "```", ""])
    if report.get("target_coincidence_prominent"):
        lines.extend(
            [
                "> **Prominent:** suspicious mode frequencies coincide with explicit `lprod_target_plan` "
                "target centers (see `target_coincidence` and chunk table).",
                "",
            ]
        )

    lines.extend(
        [
            "## Task 4 — Minimal-fix decision tree",
            "",
            "### If `PLOT_OR_DEDUPE_ARTIFACT`",
            "- Change aggregation dedupe/plotting only; regenerate catalogs/plots from retained `worker_results` where present.",
            "- No FOM rerun unless raw worker results were deleted.",
            "- ROM catalogs: re-derive deduped view from raw `modes_catalog.jsonl` (ROM compare already dedupes at load).",
            "",
            "### If `PROXY_INSENSITIVITY_SUSPECTED`",
            "- Fix `mic_output_proxy` / soundhole mask weighting in `v2_b3_mode_audio_coupling.py`.",
            "- Frequency and share fields may remain valid; recompute proxy from checkpoints where `region_dof_indices.npz` exists.",
            "- Estimate rerun: only samples needing proxy re-aggregation (not full eigen solve) if mode vectors retained.",
            "",
            "### If `FIXED_AIR_DOMAIN_SUSPECTED`",
            "- Inspect exterior air domain BC/mesh tags; validate cavity vs exterior participation on full-artifact samples.",
            "- Targeted rerun of affected frequency bands after boundary/mesh fix.",
            "",
            "### If `GEOMETRY_PROPAGATION_SUSPECTED`",
            "- Identify mesh/checkpoint reuse bug; invalidate only samples sharing duplicate mesh/checkpoint hashes.",
            "- Run extreme-case validation (Task 5) before mass rerun.",
            "",
            "### If `PHYSICALLY_PLAUSIBLE`",
            "- Document expected sensitivity from LHS geometry range; set acceptance threshold for future samples "
            "(e.g. freq should track Helmholtz estimate with |r|>0.4).",
            "- STK/ROM may still need deduped catalog and mic-proxy caveats.",
            "",
            "## Task 5 — Extreme validation specs (do not run automatically)",
            "",
            "Cheapest pre-L_prod scout/coarse commands to confirm air family shifts with geometry:",
            "",
            "```bash",
            "# 1) small body + small soundhole (near LHS minima)",
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \\",
            "  --lhs-json ROM/classic/lhs_pool.json --sample-ids sample_extreme_small \\",
            "  --execute --workers 1 --max-samples 1 \\",
            "  --geometry-override '{\"length\":0.36,\"width\":0.23,\"depth\":0.085,\"hole_radius\":0.040,\"top_thickness\":0.003}' \\",
            "  --run-scout-only",
            "",
            "# 2) large body + large soundhole (near LHS maxima)",
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \\",
            "  --lhs-json ROM/classic/lhs_pool.json --sample-ids sample_extreme_large \\",
            "  --execute --workers 1 --max-samples 1 \\",
            "  --geometry-override '{\"length\":0.57,\"width\":0.42,\"depth\":0.14,\"hole_radius\":0.050,\"top_thickness\":0.003}' \\",
            "  --run-scout-only",
            "",
            "# 3) same body, extreme hole_area/volume ratio",
            "python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_m4_production_pipeline.py \\",
            "  --lhs-json ROM/classic/lhs_pool.json --sample-ids sample_extreme_hole_ratio \\",
            "  --execute --workers 1 --max-samples 1 \\",
            "  --geometry-override '{\"length\":0.48,\"width\":0.325,\"depth\":0.10,\"hole_radius\":0.050,\"top_thickness\":0.003}' \\",
            "  --run-scout-only",
            "```",
            "",
            "**Acceptance:** if ~281 Hz family stays within ~0.01 Hz across these extremes, treat as strong artifact indication.",
            "",
            "## New FOM computation required?",
            "",
            report.get("fom_rerun_required") or "See verdict-specific tree above.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit repeated ~281/390 Hz air mode families (read-only, catalog-first).",
    )
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS))
    parser.add_argument("--samples", default="0-35", help="e.g. 0-35 or sample_018,sample_019")
    parser.add_argument("--run-id-suffix", default=DEFAULT_RUN_ID_SUFFIX)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument(
        "--full-artifact-audit",
        action="store_true",
        help="Also inspect retained mesh/checkpoint/mask for --full-samples.",
    )
    parser.add_argument(
        "--full-samples",
        default=",".join(FULL_ARTIFACT_DEFAULT_SAMPLES),
        help="Comma list for Task 2 heavy-artifact audit.",
    )
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    pool = load_lhs_pool(lhs_path)
    sample_ids = _parse_sample_ids(str(args.samples))

    catalog_rows: List[Dict[str, Any]] = []
    cluster_detail_rows: List[Dict[str, Any]] = []
    chunk_audit_rows: List[Dict[str, Any]] = []

    for sid in sample_ids:
        row = audit_sample_catalog(
            repo_root=repo_root,
            pool=pool,
            sample_id=sid,
            run_id_suffix=str(args.run_id_suffix),
        )
        catalog_rows.append(row)
        for cm in row.get("cluster_modes") or []:
            cluster_detail_rows.append(cm)

        if row.get("catalog_status") == "ok":
            run_id, _ = _resolve_run_id(pool, sid, str(args.run_id_suffix))
            run_root = _run_root(repo_root, sid, run_id)
            raw_modes, _, _ = load_fom_modes_catalog_deduped(run_root / "aggregation" / "modes_catalog.jsonl")
            chunk_audit_rows.extend(
                _chunk_target_audit_for_sample(
                    sample_id=sid,
                    run_root=run_root,
                    raw_modes=raw_modes,
                    bands=(("281", BAND_281), ("390", BAND_390)),
                )
            )

    peaks_for_discovery = [
        {
            "sample_id": r["sample_id"],
            "peak_freq_hz": r.get("peak_281_freq_hz"),
            "peak_mic": r.get("peak_281_mic"),
        }
        for r in catalog_rows
        if r.get("peak_281_freq_hz") is not None
    ]
    discovered = _discover_repeated_families(peaks_for_discovery)

    peaks_281 = [
        r
        for r in catalog_rows
        if r.get("catalog_status") == "ok" and r.get("peak_281_freq_hz") is not None
    ]
    peaks_390 = [
        r
        for r in catalog_rows
        if r.get("catalog_status") == "ok" and r.get("peak_390_freq_hz") is not None
    ]

    cross_sample = {
        "unique_geometry_fingerprints": len(
            {r.get("geometry_fingerprint") for r in catalog_rows if r.get("geometry_fingerprint")}
        ),
        "cluster_281hz": _family_stats(
            [{"frequency_hz": r.get("peak_281_freq_hz"), "mic_output_proxy": r.get("peak_281_mic")} for r in peaks_281]
        ),
        "cluster_390hz": _family_stats(
            [{"frequency_hz": r.get("peak_390_freq_hz"), "mic_output_proxy": r.get("peak_390_mic")} for r in peaks_390]
        ),
        "total_exact_duplicate_groups_281": sum(
            int(r.get("band_281_exact_duplicate_groups") or 0) for r in catalog_rows
        ),
        "total_exact_duplicate_groups_390": sum(
            int(r.get("band_390_exact_duplicate_groups") or 0) for r in catalog_rows
        ),
    }
    c281 = cross_sample.get("cluster_281hz") or {}
    if c281.get("mic_mean") and c281.get("mic_range") is not None:
        c281["mic_rel_span"] = (
            float(c281["mic_range"]) / float(c281["mic_mean"]) if float(c281["mic_mean"]) > 0 else None
        )

    correlations = {
        "band_281": _correlation_block(catalog_rows, band_key="281"),
        "band_390": _correlation_block(catalog_rows, band_key="390"),
    }

    full_rows: List[Dict[str, Any]] = []
    if args.full_artifact_audit:
        for sid in _parse_sample_ids(str(args.full_samples)):
            full_rows.append(
                audit_full_artifacts(
                    repo_root=repo_root,
                    pool=pool,
                    sample_id=sid,
                    run_id_suffix=str(args.run_id_suffix),
                )
            )

    target_coincidence = _target_coincidence_summary(chunk_audit_rows)
    prominent_target = any(
        (target_coincidence.get(k) or {}).get("modes_with_target_within_0p1_hz", 0) >= 8
        for k in ("281", "390")
    )

    report: Dict[str, Any] = {
        "schema": "m4_repeated_air_mode_family_audit_v1",
        "lhs_json": rel(lhs_path, repo_root=repo_root),
        "sample_ids": sample_ids,
        "dedupe_tolerance_hz": ROM_DEDUPE_TOLERANCE_HZ,
        "per_sample": catalog_rows,
        "cluster_detail": cluster_detail_rows,
        "discovered_families": discovered,
        "cross_sample": cross_sample,
        "correlations": correlations,
        "sample_000_comparison": _compare_sample_000(catalog_rows),
        "chunk_target_audit": chunk_audit_rows,
        "target_coincidence": target_coincidence,
        "target_coincidence_prominent": prominent_target,
        "full_artifact_rows": full_rows,
        "full_artifact_reuse": _geometry_reuse_flags(full_rows) if full_rows else {},
    }
    report["verdict"] = _determine_verdict(report)

    if report["verdict"] in ("PLOT_OR_DEDUPE_ARTIFACT", "PROXY_INSENSITIVITY_SUSPECTED"):
        report["fom_rerun_required"] = (
            "No full FOM rerun if raw worker_results/mode vectors retained; "
            "re-aggregation and/or proxy recompute may suffice."
        )
    elif report["verdict"] == "GEOMETRY_PROPAGATION_SUSPECTED":
        report["fom_rerun_required"] = "Partial rerun for samples with duplicate mesh/checkpoint hashes after bugfix."
    elif report["verdict"] == "PHYSICALLY_PLAUSIBLE":
        report["fom_rerun_required"] = "No FOM rerun required for investigation; optional extreme validation (Task 5) only."
    else:
        report["fom_rerun_required"] = "Run Task 2 full-artifact audit on VM; then Task 5 extreme scouts if still inconclusive."

    # Terminal summary
    print(f"audited_samples={len(catalog_rows)}")
    print(f"verdict={report['verdict']}")
    print(f"unique_geometry_fingerprints={cross_sample.get('unique_geometry_fingerprints')}")
    if c281:
        print(
            f"cluster_281hz: n={c281.get('sample_count')} "
            f"freq_range={c281.get('frequency_range_hz'):.6f} Hz "
            f"mic_range={c281.get('mic_range'):.6e} "
            f"exact_mic_10dp={c281.get('exact_mic_repeats_10dp')}"
        )
    c390 = cross_sample.get("cluster_390hz") or {}
    if c390:
        print(
            f"cluster_390hz: n={c390.get('sample_count')} "
            f"freq_range={c390.get('frequency_range_hz'):.6f} Hz"
        )
    if prominent_target:
        print("WARNING: mode frequencies coincide with explicit lprod target centers near 281/390 Hz")
    if full_rows:
        reuse = report.get("full_artifact_reuse") or {}
        if reuse.get("duplicate_mesh_sha256_groups"):
            print(f"WARNING: duplicate_mesh_sha256_groups={reuse['duplicate_mesh_sha256_groups']}")

    for r in catalog_rows:
        if r.get("catalog_status") != "ok":
            print(f"  {r['sample_id']}: catalog missing")
            continue
        print(
            f"  {r['sample_id']}: f281={r.get('peak_281_freq_hz')} mic={r.get('peak_281_mic')} "
            f"air={r.get('peak_281_air_share')} dups281={r.get('band_281_exact_duplicate_groups')} "
            f"targets281={r.get('lprod_targets_near_281_hz')}"
        )

    def _write_out(path: Path, content: str) -> None:
        out = path if path.is_absolute() else repo_root / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        print(f"wrote {rel(out, repo_root=repo_root)}")

    json_out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
    _write_out(json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")

    md_out = args.md_out if args.md_out.is_absolute() else repo_root / args.md_out
    _write_out(md_out, _render_markdown(report) + "\n")

    csv_out = args.csv_out if args.csv_out.is_absolute() else repo_root / args.csv_out
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CATALOG_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in cluster_detail_rows:
            flat = dict(row)
            if flat.get("provenance_chunk_ids") is not None:
                flat["provenance_chunk_ids"] = ",".join(flat["provenance_chunk_ids"])
            if flat.get("chunk_interval_hz") is not None:
                flat["chunk_interval_hz"] = json.dumps(flat["chunk_interval_hz"])
            writer.writerow(flat)
    print(f"wrote {rel(csv_out, repo_root=repo_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
