#!/usr/bin/env python3
"""M4 ROM pre-prediction and ROM/FOM frequency comparison (no legacy FOM rerun)."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_lhs_pool_bridge import (  # noqa: E402
    AGG_PASS,
    DEFAULT_RUN_ID_SUFFIX,
    lhs_entry_index,
    load_lhs_pool,
    read_run_production_summary,
    sync_lhs_pool_entry,
    write_lhs_pool_with_backup,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

ROM_PREDICTION_SCHEMA = "rom_prediction_pre_fom_v1"
ROM_FOM_COMPARISON_SCHEMA = "rom_fom_comparison_v2"
ROM_INDEX_SCHEMA = "rom_fom_comparison_index_v1"
ACCURACY_HISTORY_SCHEMA = "rom_accuracy_history_v1"
ACCURACY_SUMMARY_SCHEMA = "rom_accuracy_summary_v1"

DEFAULT_MAX_MATCH_DISTANCE_HZ = 15.0
DEFAULT_ROM_NEV = 0
MATCHING_METHOD = "greedy_nearest_hz_one_to_one"

ACCURACY_BAND_HZ: Tuple[float, float] = (60.0, 550.0)
PRIMARY_ACCURACY_METRIC = "median_relative_error"
TARGET_MEDIAN_RELATIVE_ERROR = 0.05
TOP_RADIATION_MODE_FRACTION = 0.20

ACCURACY_HISTORY_CSV = "rom_accuracy_history.csv"
ACCURACY_SUMMARY_JSON = "rom_accuracy_summary.json"

ROM_STATUS_COMPLETED = "COMPLETED"
ROM_STATUS_FAILED = "FAILED"
ROM_STATUS_SKIPPED = "SKIPPED"

ROM_COMPARE_COMPLETED = "COMPLETED"
ROM_COMPARE_FAILED = "FAILED"
ROM_COMPARE_NOT_AVAILABLE = "NOT_AVAILABLE"

LIGHTWEIGHT_MODE_FIELDS = (
    "frequency_hz",
    "top_share",
    "back_share",
    "air_share",
    "coupling_class",
    "bridge_excitation_coupling",
    "mic_output_proxy",
    "radiation_proxy",
    "modal_norm",
)


def _repo_root_from_script(script_dir: Path) -> Path:
    return detect_repo_root(script_dir)


def _import_rom_manager(repo_root: Path):
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from FEM.rom import ROMManager  # noqa: WPS433

    return ROMManager(base_dir=repo_root)


def _import_surrogate_helpers():
    from v2_b3_m4_modal_surrogate_lib import (  # noqa: WPS433
        load_surrogate_model,
        predict_modal_frequencies,
        resolve_active_rom_backend,
        surrogate_json_path,
    )

    return load_surrogate_model, predict_modal_frequencies, resolve_active_rom_backend, surrogate_json_path


def rom_dir_for_run(run_root: Path) -> Path:
    return run_root / "rom"


def rom_prediction_path(run_root: Path) -> Path:
    return rom_dir_for_run(run_root) / "rom_prediction_pre_fom.json"


def rom_comparison_path(run_root: Path) -> Path:
    return rom_dir_for_run(run_root) / "rom_fom_comparison.json"


def comparisons_project_dir(repo_root: Path, shape_name: str) -> Path:
    return repo_root / "ROM" / shape_name / "comparisons"


def comparisons_index_path(repo_root: Path, shape_name: str) -> Path:
    return comparisons_project_dir(repo_root, shape_name) / "rom_fom_comparison_index.jsonl"


def project_comparison_copy_path(
    repo_root: Path, *, shape_name: str, sample_id: str, run_id: str
) -> Path:
    name = f"{sample_id}__{run_id}_rom_fom_comparison.json"
    return comparisons_project_dir(repo_root, shape_name) / name


def accuracy_history_path(repo_root: Path, shape_name: str) -> Path:
    return comparisons_project_dir(repo_root, shape_name) / ACCURACY_HISTORY_CSV


def accuracy_summary_path(repo_root: Path, shape_name: str) -> Path:
    return comparisons_project_dir(repo_root, shape_name) / ACCURACY_SUMMARY_JSON


def _in_accuracy_band(f_hz: float, band: Tuple[float, float] = ACCURACY_BAND_HZ) -> bool:
    lo, hi = band
    return lo <= float(f_hz) <= hi


def filter_rom_frequencies_to_band(
    frequencies_hz: Sequence[float],
    *,
    band: Tuple[float, float] = ACCURACY_BAND_HZ,
) -> List[float]:
    return sorted(float(f) for f in frequencies_hz if _in_accuracy_band(f, band))


def filter_fom_modes_to_band(
    modes: Sequence[Mapping[str, Any]],
    *,
    band: Tuple[float, float] = ACCURACY_BAND_HZ,
) -> List[Dict[str, Any]]:
    out = [dict(m) for m in modes if _in_accuracy_band(float(m["frequency_hz"]), band)]
    out.sort(key=lambda r: float(r["frequency_hz"]))
    return out


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    pairs = [
        (float(v), float(w))
        for v, w in zip(values, weights)
        if v == v and w == w and float(w) > 0.0
    ]
    if not pairs:
        return None
    num = sum(v * w for v, w in pairs)
    den = sum(w for _, w in pairs)
    return num / den if den > 0 else None


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    pairs = sorted(
        [(float(v), float(w)) for v, w in zip(values, weights) if v == v and w == w and float(w) > 0.0],
        key=lambda t: t[0],
    )
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    half = total / 2.0
    acc = 0.0
    for val, wt in pairs:
        acc += wt
        if acc >= half:
            return val
    return pairs[-1][0]


def _training_sample_count_from_prediction(rom_prediction: Mapping[str, Any]) -> Optional[int]:
    for key in ("training_sample_count_at_prediction", "num_basis_modes"):
        val = rom_prediction.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    raw = rom_prediction.get("raw")
    if isinstance(raw, Mapping):
        for key in ("training_sample_count", "k_neighbors_used"):
            val = raw.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
    return None


def load_fom_modes_catalog(catalog_path: Path) -> List[Dict[str, Any]]:
    if not catalog_path.is_file():
        raise FileNotFoundError(f"FOM modes catalog missing: {catalog_path}")
    modes: List[Dict[str, Any]] = []
    with catalog_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{catalog_path}:{line_no}: invalid JSONL: {exc}") from exc
            if rec.get("frequency_hz") is None:
                continue
            try:
                f_hz = float(rec["frequency_hz"])
            except (TypeError, ValueError):
                continue
            if not (f_hz == f_hz and f_hz > 0.0):
                continue
            modes.append(dict(rec))
    modes.sort(key=lambda r: float(r["frequency_hz"]))
    return modes


def _unavailable_mode_record(frequency_hz: float) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"frequency_hz": round(float(frequency_hz), 6)}
    availability: Dict[str, bool] = {"frequency_hz": True}
    for field in LIGHTWEIGHT_MODE_FIELDS:
        if field == "frequency_hz":
            continue
        rec[field] = None
        availability[field] = False
    rec["field_availability"] = availability
    return rec


def greedy_nearest_hz_match(
    *,
    rom_frequencies_hz: Sequence[float],
    fom_modes: Sequence[Mapping[str, Any]],
    max_match_distance_hz: Optional[float] = DEFAULT_MAX_MATCH_DISTANCE_HZ,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    One-to-one greedy matching: walk FOM ascending, assign nearest unused ROM frequency.
    """
    rom_list = [float(f) for f in rom_frequencies_hz]
    rom_used = [False] * len(rom_list)
    matches: List[Dict[str, Any]] = []

    for fom_rec in sorted(fom_modes, key=lambda r: float(r["frequency_hz"])):
        fom_hz = float(fom_rec["frequency_hz"])
        best_j: Optional[int] = None
        best_dist = float("inf")
        for j, rom_hz in enumerate(rom_list):
            if rom_used[j]:
                continue
            dist = abs(rom_hz - fom_hz)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is None:
            continue
        if max_match_distance_hz is not None and best_dist > float(max_match_distance_hz):
            continue
        rom_used[best_j] = True
        rom_hz = rom_list[best_j]
        rel_err = abs(rom_hz - fom_hz) / fom_hz if abs(fom_hz) > 1e-12 else float("nan")
        matches.append(
            {
                "rom_frequency_hz": round(rom_hz, 6),
                "fom_frequency_hz": round(fom_hz, 6),
                "abs_error_hz": round(best_dist, 6),
                "relative_error": round(rel_err, 8) if rel_err == rel_err else None,
                "fom_coupling_class": fom_rec.get("coupling_class"),
                "fom_radiation_proxy": fom_rec.get("radiation_proxy"),
                "fom_mic_output_proxy": fom_rec.get("mic_output_proxy"),
            }
        )

    unmatched_rom = sum(1 for u in rom_used if not u)
    unmatched_fom = len(fom_modes) - len(matches)
    meta = {
        "method": MATCHING_METHOD,
        "max_match_distance_hz": max_match_distance_hz,
        "matched_mode_count": len(matches),
        "unmatched_rom_count": unmatched_rom,
        "unmatched_fom_count": unmatched_fom,
        "rom_mode_count": len(rom_list),
        "fom_mode_count": len(fom_modes),
    }
    return matches, meta


def _error_metrics(matches: Sequence[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    if not matches:
        return {
            "mean_abs_error_hz": None,
            "median_abs_error_hz": None,
            "max_abs_error_hz": None,
            "mean_relative_error": None,
            "median_relative_error": None,
            "p90_relative_error": None,
            "max_relative_error": None,
            "audio_weighted_mean_relative_error": None,
            "audio_weighted_median_relative_error": None,
            "top_radiation_modes_mean_relative_error": None,
            "top_radiation_modes_median_relative_error": None,
            "top_radiation_modes_matched_count": 0,
        }
    abs_errs = [float(m["abs_error_hz"]) for m in matches]
    rel_errs = [
        float(m["relative_error"])
        for m in matches
        if m.get("relative_error") is not None and m["relative_error"] == m["relative_error"]
    ]
    rad_weights = []
    for m in matches:
        rp = m.get("fom_radiation_proxy")
        try:
            w = float(rp) if rp is not None else 0.0
        except (TypeError, ValueError):
            w = 0.0
        rad_weights.append(max(w, 0.0))

    audio_mean = _weighted_mean(rel_errs, rad_weights)
    audio_median = _weighted_median(rel_errs, rad_weights)

    top_n = max(1, int(len(matches) * TOP_RADIATION_MODE_FRACTION + 0.999))
    top_matches = sorted(
        matches,
        key=lambda m: float(m.get("fom_radiation_proxy") or 0.0),
        reverse=True,
    )[:top_n]
    top_rel = [
        float(m["relative_error"])
        for m in top_matches
        if m.get("relative_error") is not None and m["relative_error"] == m["relative_error"]
    ]

    p90 = _percentile(rel_errs, 90.0)
    return {
        "mean_abs_error_hz": round(statistics.mean(abs_errs), 6),
        "median_abs_error_hz": round(statistics.median(abs_errs), 6),
        "max_abs_error_hz": round(max(abs_errs), 6),
        "mean_relative_error": round(statistics.mean(rel_errs), 8) if rel_errs else None,
        "median_relative_error": round(statistics.median(rel_errs), 8) if rel_errs else None,
        "p90_relative_error": round(p90, 8) if p90 is not None and p90 == p90 else None,
        "max_relative_error": round(max(rel_errs), 8) if rel_errs else None,
        "audio_weighted_mean_relative_error": round(audio_mean, 8) if audio_mean is not None else None,
        "audio_weighted_median_relative_error": round(audio_median, 8)
        if audio_median is not None
        else None,
        "top_radiation_modes_mean_relative_error": round(statistics.mean(top_rel), 8)
        if top_rel
        else None,
        "top_radiation_modes_median_relative_error": round(statistics.median(top_rel), 8)
        if top_rel
        else None,
        "top_radiation_modes_matched_count": len(top_rel),
    }


def _accuracy_spec_block(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    median_rel = metrics.get("median_relative_error")
    meets = (
        median_rel is not None
        and median_rel == median_rel
        and float(median_rel) <= TARGET_MEDIAN_RELATIVE_ERROR
    )
    return {
        "frequency_band_hz": list(ACCURACY_BAND_HZ),
        "primary_metric": PRIMARY_ACCURACY_METRIC,
        "target_median_relative_error": TARGET_MEDIAN_RELATIVE_ERROR,
        "meets_target": bool(meets) if median_rel is not None else False,
        "secondary_metrics": [
            "mean_relative_error",
            "p90_relative_error",
            "median_abs_error_hz",
            "mean_abs_error_hz",
            "max_abs_error_hz",
            "matched_mode_count",
            "unmatched_rom_count",
            "unmatched_fom_count",
        ],
        "diagnostic_metrics": ["max_relative_error"],
    }


def resolve_sample_context(
    *,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id: str,
    run_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    shape_name = str(pool.get("shape_name") or "classic")
    idx = lhs_entry_index(pool, sample_id)
    if idx is None:
        raise KeyError(f"sample_id {sample_id!r} not found in LHS pool")
    entry = (pool.get("entries") or [])[idx]
    params = dict(entry.get("parameters") or {})

    if run_root is None:
        guitars = (
            (repo_root or _repo_root_from_script(Path(__file__).resolve().parent))
            / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
        )
        run_root = guitars / sample_id / "runs" / run_id

    sample_input_path = run_root / "sample" / "sample_input.json"
    if sample_input_path.is_file():
        try:
            si = load_json(sample_input_path)
            if isinstance(si.get("parameters"), dict):
                params = dict(si["parameters"])
            if si.get("shape_name"):
                shape_name = str(si["shape_name"])
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    return {
        "sample_id": sample_id,
        "lhs_row_index": int(idx),
        "shape_name": shape_name,
        "parameters": params,
        "run_id": run_id,
        "run_root": run_root.resolve(),
    }


def _run_m4_surrogate_prediction(
    *,
    repo_root: Path,
    shape_name: str,
    parameters: Mapping[str, Any],
    nev: int,
) -> Dict[str, Any]:
    load_surrogate_model, predict_modal_frequencies, _, surrogate_json_path = _import_surrogate_helpers()
    t0 = time.perf_counter()
    model = load_surrogate_model(repo_root, shape_name)
    out = predict_modal_frequencies(model, parameters, nev=int(nev))
    elapsed = time.perf_counter() - t0
    freqs = [float(f) for f in (out.get("frequencies_hz") or [])]
    modes = [_unavailable_mode_record(f) for f in freqs]
    return {
        "status": ROM_STATUS_COMPLETED,
        "method": str(out.get("method") or "m4_modal_surrogate"),
        "source": str(surrogate_json_path(repo_root, shape_name)),
        "confidence": "m4_fom_knn_surrogate",
        "runtime_s": round(elapsed, 4),
        "nev_requested": int(nev),
        "nev_returned": int(out.get("nev_returned") or len(freqs)),
        "num_basis_modes": int(model.get("training_sample_count") or 0),
        "frequencies_hz": [round(f, 6) for f in freqs],
        "modes": modes,
        "error": None,
        "raw": out,
    }


def _run_legacy_basis_prediction(
    *,
    repo_root: Path,
    shape_name: str,
    parameters: Mapping[str, Any],
    nev: int,
) -> Dict[str, Any]:
    manager = _import_rom_manager(repo_root)
    t0 = time.perf_counter()
    result = manager.solve_online(shape_name, dict(parameters), nev=int(nev))
    elapsed = time.perf_counter() - t0
    freqs = [float(f) for f in (result.get("freqs_hz") or [])]
    modes = [_unavailable_mode_record(f) for f in freqs]
    return {
        "status": ROM_STATUS_COMPLETED,
        "method": "ROMManager.solve_online",
        "source": str(result.get("basis_path") or ""),
        "confidence": "reduced_basis_online",
        "runtime_s": round(elapsed, 4),
        "nev_requested": int(nev),
        "nev_returned": int(result.get("nev") or len(freqs)),
        "num_basis_modes": int(result.get("num_basis_modes") or 0),
        "frequencies_hz": [round(f, 6) for f in freqs],
        "modes": modes,
        "error": None,
        "raw": {
            "elapsed_s": result.get("elapsed_s"),
            "basis_path": result.get("basis_path"),
        },
    }


def run_rom_online_prediction(
    *,
    repo_root: Path,
    shape_name: str,
    parameters: Mapping[str, Any],
    nev: int = DEFAULT_ROM_NEV,
) -> Dict[str, Any]:
    """
    Predict modal frequencies using the best available ROM backend:
    1) M4 modal surrogate trained from modes_catalog.jsonl (preferred)
    2) Legacy POD reduced_basis.npz + operator projection (optional fallback)
    """
    _, _, resolve_active_rom_backend, _ = _import_surrogate_helpers()
    backend = resolve_active_rom_backend(repo_root, shape_name)
    t0 = time.perf_counter()
    try:
        if backend == "m4_surrogate":
            return _run_m4_surrogate_prediction(
                repo_root=repo_root,
                shape_name=shape_name,
                parameters=parameters,
                nev=nev,
            )
        if backend == "legacy_basis":
            return _run_legacy_basis_prediction(
                repo_root=repo_root,
                shape_name=shape_name,
                parameters=parameters,
                nev=nev,
            )
        raise FileNotFoundError(
            f"No ROM model for shape {shape_name!r}. "
            f"Build M4 surrogate: build_m4_rom_from_completed_fom.py "
            f"or legacy basis: FEM/scripts/rom_pipeline.py build-basis --shape {shape_name}"
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "status": ROM_STATUS_FAILED,
            "method": backend if backend != "none" else "rom_unavailable",
            "source": None,
            "confidence": None,
            "runtime_s": round(elapsed, 4),
            "nev_requested": int(nev),
            "nev_returned": 0,
            "num_basis_modes": 0,
            "frequencies_hz": [],
            "modes": [],
            "error": str(exc),
            "raw": None,
        }


def build_rom_prediction_document(
    *,
    context: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "schema": ROM_PREDICTION_SCHEMA,
        "generated_utc": utc_now(),
        "sample_id": context["sample_id"],
        "lhs_row_index": context["lhs_row_index"],
        "shape_name": context["shape_name"],
        "parameters": dict(context.get("parameters") or {}),
        "run_id": context.get("run_id"),
        "status": prediction.get("status"),
        "method": prediction.get("method"),
        "source": prediction.get("source"),
        "confidence": prediction.get("confidence"),
        "runtime_s": prediction.get("runtime_s"),
        "nev_requested": prediction.get("nev_requested"),
        "nev_returned": prediction.get("nev_returned"),
        "num_basis_modes": prediction.get("num_basis_modes"),
        "training_sample_count_at_prediction": _training_sample_count_from_prediction(prediction),
        "frequencies_hz": list(prediction.get("frequencies_hz") or []),
        "modes": list(prediction.get("modes") or []),
        "error": prediction.get("error"),
    }


def write_rom_prediction_pre_fom(run_root: Path, document: Mapping[str, Any]) -> Path:
    out_dir = rom_dir_for_run(run_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = rom_prediction_path(run_root)
    write_json_atomic(path, dict(document))
    return path


def load_rom_prediction_pre_fom(run_root: Path) -> Optional[Dict[str, Any]]:
    path = rom_prediction_path(run_root)
    if not path.is_file():
        return None
    try:
        return load_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def build_rom_fom_comparison(
    *,
    context: Mapping[str, Any],
    rom_prediction: Mapping[str, Any],
    fom_summary: Mapping[str, Any],
    fom_modes: Sequence[Mapping[str, Any]],
    max_match_distance_hz: Optional[float] = DEFAULT_MAX_MATCH_DISTANCE_HZ,
    rom_prediction_path_rel: Optional[str] = None,
    fom_catalog_path_rel: Optional[str] = None,
    accuracy_band_hz: Tuple[float, float] = ACCURACY_BAND_HZ,
) -> Dict[str, Any]:
    rom_freqs_all = list(rom_prediction.get("frequencies_hz") or [])
    fom_modes_all = list(fom_modes)
    rom_freqs = filter_rom_frequencies_to_band(rom_freqs_all, band=accuracy_band_hz)
    fom_modes_band = filter_fom_modes_to_band(fom_modes_all, band=accuracy_band_hz)
    matches, match_meta = greedy_nearest_hz_match(
        rom_frequencies_hz=rom_freqs,
        fom_modes=fom_modes_band,
        max_match_distance_hz=max_match_distance_hz,
    )
    metrics = _error_metrics(matches)
    training_count = _training_sample_count_from_prediction(rom_prediction)
    accuracy_spec = _accuracy_spec_block(metrics)

    fom_runtime = None
    rt_path_hint = fom_summary.get("runtime_summary_path")
    if isinstance(rt_path_hint, str):
        try:
            rt = load_json(Path(rt_path_hint))
            fom_runtime = rt.get("total_wall_s") or rt.get("elapsed_s")
        except (OSError, ValueError, json.JSONDecodeError, TypeError):
            fom_runtime = None
    if fom_runtime is None:
        fom_runtime = fom_summary.get("stage_wall_times_s", {}).get("total") if isinstance(
            fom_summary.get("stage_wall_times_s"), dict
        ) else None

    rom_status = str(rom_prediction.get("status") or ROM_STATUS_FAILED)
    if rom_status == ROM_STATUS_FAILED:
        compare_status = ROM_COMPARE_FAILED
    elif rom_status == ROM_STATUS_COMPLETED:
        compare_status = ROM_COMPARE_COMPLETED
    else:
        compare_status = ROM_COMPARE_NOT_AVAILABLE

    return {
        "schema": ROM_FOM_COMPARISON_SCHEMA,
        "generated_utc": utc_now(),
        "sample_id": context["sample_id"],
        "lhs_row_index": context["lhs_row_index"],
        "run_id": context["run_id"],
        "shape_name": context["shape_name"],
        "parameters": dict(context.get("parameters") or {}),
        "accuracy_spec": accuracy_spec,
        "frequency_band_hz": list(accuracy_band_hz),
        "training_sample_count_at_prediction": training_count,
        "rom_prediction_status": rom_status,
        "rom_prediction_method": rom_prediction.get("method"),
        "rom_prediction_path": rom_prediction_path_rel,
        "rom_runtime_s": rom_prediction.get("runtime_s"),
        "fom_run_status": fom_summary.get("terminal_status"),
        "fom_aggregation_status": fom_summary.get("aggregation_status"),
        "fom_deduped_mode_count": fom_summary.get("deduped_modes") or len(fom_modes_all),
        "fom_deduped_mode_count_in_band": len(fom_modes_band),
        "fom_catalog_path": fom_catalog_path_rel,
        "fom_runtime_s": fom_runtime,
        "rom_mode_count": match_meta["rom_mode_count"],
        "rom_mode_count_total": len(rom_freqs_all),
        "fom_mode_count_in_band": match_meta["fom_mode_count"],
        "frequency_matching_method": match_meta["method"],
        "max_match_distance_hz": match_meta["max_match_distance_hz"],
        "matched_mode_count": match_meta["matched_mode_count"],
        "unmatched_rom_count": match_meta["unmatched_rom_count"],
        "unmatched_fom_count": match_meta["unmatched_fom_count"],
        **metrics,
        "per_mode_matches": matches,
        "rom_frequencies_hz": rom_freqs,
        "rom_frequencies_hz_total": [round(float(f), 6) for f in rom_freqs_all],
        "fom_frequencies_hz": [round(float(r["frequency_hz"]), 6) for r in fom_modes_band],
        "fom_frequencies_hz_total": [round(float(r["frequency_hz"]), 6) for r in fom_modes_all],
        "status": compare_status,
        "warnings": [],
    }


def write_rom_fom_comparison_artifacts(
    *,
    repo_root: Path,
    run_root: Path,
    comparison: Mapping[str, Any],
    copy_to_project: bool = True,
    write_csv: bool = False,
) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    rom_dir = rom_dir_for_run(run_root)
    rom_dir.mkdir(parents=True, exist_ok=True)

    primary = rom_comparison_path(run_root)
    write_json_atomic(primary, dict(comparison))
    out["run_comparison"] = primary

    shape = str(comparison.get("shape_name") or "classic")
    sid = str(comparison["sample_id"])
    rid = str(comparison["run_id"])

    if copy_to_project:
        proj_dir = comparisons_project_dir(repo_root, shape)
        proj_dir.mkdir(parents=True, exist_ok=True)
        copy_path = project_comparison_copy_path(repo_root, shape_name=shape, sample_id=sid, run_id=rid)
        write_json_atomic(copy_path, dict(comparison))
        out["project_comparison"] = copy_path

        index_path = comparisons_index_path(repo_root, shape)
        index_row = {
            "schema": ROM_INDEX_SCHEMA,
            "recorded_utc": utc_now(),
            "sample_id": sid,
            "run_id": rid,
            "lhs_row_index": comparison.get("lhs_row_index"),
            "comparison_path": rel(copy_path, repo_root=repo_root),
            "run_comparison_path": rel(primary, repo_root=repo_root),
            "status": comparison.get("status"),
            "matched_mode_count": comparison.get("matched_mode_count"),
            "median_relative_error": comparison.get("median_relative_error"),
            "mean_relative_error": comparison.get("mean_relative_error"),
            "p90_relative_error": comparison.get("p90_relative_error"),
            "median_abs_error_hz": comparison.get("median_abs_error_hz"),
            "mean_abs_error_hz": comparison.get("mean_abs_error_hz"),
            "training_sample_count_at_prediction": comparison.get(
                "training_sample_count_at_prediction"
            ),
            "meets_target": (comparison.get("accuracy_spec") or {}).get("meets_target"),
        }
        with index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(index_row, sort_keys=True) + "\n")
        out["index"] = index_path

        hist_path, summary_path = update_accuracy_history(
            repo_root=repo_root,
            shape_name=shape,
            comparison=comparison,
        )
        out["accuracy_history"] = hist_path
        out["accuracy_summary"] = summary_path

    if write_csv:
        csv_path = (
            comparisons_project_dir(repo_root, shape) / f"{sid}__{rid}_rom_fom_comparison.csv"
            if copy_to_project
            else rom_dir / "rom_fom_comparison.csv"
        )
        _write_comparison_csv(csv_path, comparison)
        out["csv"] = csv_path

    return out


def _write_comparison_csv(path: Path, comparison: Mapping[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "lhs_row_index",
        "run_id",
        "rom_frequency_hz",
        "fom_frequency_hz",
        "abs_error_hz",
        "relative_error",
        "fom_coupling_class",
        "fom_radiation_proxy",
        "fom_mic_output_proxy",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        base = {
            "sample_id": comparison.get("sample_id"),
            "lhs_row_index": comparison.get("lhs_row_index"),
            "run_id": comparison.get("run_id"),
        }
        for row in comparison.get("per_mode_matches") or []:
            writer.writerow({**base, **row})


ACCURACY_HISTORY_FIELDS: Tuple[str, ...] = (
    "recorded_utc",
    "sample_id",
    "run_id",
    "training_sample_count_at_prediction",
    "rom_prediction_method",
    "matched_mode_count",
    "median_relative_error",
    "mean_relative_error",
    "p90_relative_error",
    "median_abs_error_hz",
    "mean_abs_error_hz",
    "meets_target",
    "status",
)


def _comparison_to_history_row(comparison: Mapping[str, Any]) -> Dict[str, Any]:
    spec = comparison.get("accuracy_spec") or {}
    return {
        "recorded_utc": comparison.get("generated_utc") or utc_now(),
        "sample_id": comparison.get("sample_id"),
        "run_id": comparison.get("run_id"),
        "training_sample_count_at_prediction": comparison.get("training_sample_count_at_prediction"),
        "rom_prediction_method": comparison.get("rom_prediction_method"),
        "matched_mode_count": comparison.get("matched_mode_count"),
        "median_relative_error": comparison.get("median_relative_error"),
        "mean_relative_error": comparison.get("mean_relative_error"),
        "p90_relative_error": comparison.get("p90_relative_error"),
        "median_abs_error_hz": comparison.get("median_abs_error_hz"),
        "mean_abs_error_hz": comparison.get("mean_abs_error_hz"),
        "meets_target": spec.get("meets_target"),
        "status": comparison.get("status"),
    }


def _read_accuracy_history_csv(path: Path) -> List[Dict[str, Any]]:
    import csv

    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def _write_accuracy_history_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ACCURACY_HISTORY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in ACCURACY_HISTORY_FIELDS})


def _coerce_history_numeric(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for key in (
        "training_sample_count_at_prediction",
        "matched_mode_count",
        "median_relative_error",
        "mean_relative_error",
        "p90_relative_error",
        "median_abs_error_hz",
        "mean_abs_error_hz",
    ):
        val = out.get(key)
        if val in (None, "", "None"):
            out[key] = None
            continue
        try:
            out[key] = float(val) if "error" in key or "hz" in key else int(float(val))
        except (TypeError, ValueError):
            out[key] = None
    meets = out.get("meets_target")
    if isinstance(meets, str):
        out["meets_target"] = meets.strip().lower() in ("true", "1", "yes")
    return out


def build_accuracy_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    parsed = [_coerce_history_numeric(r) for r in rows if r.get("status") == ROM_COMPARE_COMPLETED]
    medians = [
        float(r["median_relative_error"])
        for r in parsed
        if r.get("median_relative_error") is not None and r["median_relative_error"] == r["median_relative_error"]
    ]
    meets = [bool(r.get("meets_target")) for r in parsed if r.get("meets_target") is not None]

    by_training: Dict[int, List[float]] = {}
    for r in parsed:
        tc = r.get("training_sample_count_at_prediction")
        med = r.get("median_relative_error")
        if tc is None or med is None or med != med:
            continue
        by_training.setdefault(int(tc), []).append(float(med))

    training_breakdown = []
    for tc in sorted(by_training):
        vals = by_training[tc]
        training_breakdown.append(
            {
                "training_sample_count_at_prediction": tc,
                "comparison_count": len(vals),
                "median_of_median_relative_error": round(statistics.median(vals), 8),
                "mean_of_median_relative_error": round(statistics.mean(vals), 8),
                "samples_meeting_target": sum(1 for v in vals if v <= TARGET_MEDIAN_RELATIVE_ERROR),
            }
        )

    latest = parsed[-1] if parsed else {}
    return {
        "schema": ACCURACY_SUMMARY_SCHEMA,
        "generated_utc": utc_now(),
        "comparison_count": len(parsed),
        "frequency_band_hz": list(ACCURACY_BAND_HZ),
        "primary_metric": PRIMARY_ACCURACY_METRIC,
        "target_median_relative_error": TARGET_MEDIAN_RELATIVE_ERROR,
        "aggregate": {
            "median_of_median_relative_error": round(statistics.median(medians), 8) if medians else None,
            "mean_of_median_relative_error": round(statistics.mean(medians), 8) if medians else None,
            "p90_of_median_relative_error": round(_percentile(medians, 90.0), 8)
            if medians
            else None,
            "samples_meeting_target": sum(1 for v in medians if v <= TARGET_MEDIAN_RELATIVE_ERROR),
            "samples_meeting_target_fraction": round(
                sum(1 for v in medians if v <= TARGET_MEDIAN_RELATIVE_ERROR) / len(medians), 4
            )
            if medians
            else None,
            "latest_sample_id": latest.get("sample_id"),
            "latest_run_id": latest.get("run_id"),
            "latest_median_relative_error": latest.get("median_relative_error"),
            "latest_training_sample_count_at_prediction": latest.get(
                "training_sample_count_at_prediction"
            ),
            "latest_meets_target": latest.get("meets_target"),
        },
        "by_training_sample_count": training_breakdown,
    }


def update_accuracy_history(
    *,
    repo_root: Path,
    shape_name: str,
    comparison: Mapping[str, Any],
) -> Tuple[Path, Path]:
    """Append or replace history row for sample/run; rebuild rolling summary."""
    hist_path = accuracy_history_path(repo_root, shape_name)
    summary_path = accuracy_summary_path(repo_root, shape_name)
    new_row = _comparison_to_history_row(comparison)
    rows = _read_accuracy_history_csv(hist_path)
    key = (str(new_row.get("sample_id")), str(new_row.get("run_id")))
    rows = [r for r in rows if (str(r.get("sample_id")), str(r.get("run_id"))) != key]
    rows.append(new_row)
    rows.sort(key=lambda r: str(r.get("recorded_utc") or ""))
    _write_accuracy_history_csv(hist_path, rows)
    write_json_atomic(summary_path, build_accuracy_summary(rows))
    return hist_path, summary_path


def lhs_pool_rom_patch_from_comparison(comparison: Mapping[str, Any]) -> Dict[str, Any]:
    status = str(comparison.get("status") or ROM_COMPARE_NOT_AVAILABLE)
    rom_pred = str(comparison.get("rom_prediction_status") or ROM_STATUS_SKIPPED)
    if status == ROM_COMPARE_COMPLETED:
        cmp_status = ROM_COMPARE_COMPLETED
    elif status == ROM_COMPARE_FAILED:
        cmp_status = ROM_COMPARE_FAILED
    else:
        cmp_status = ROM_COMPARE_NOT_AVAILABLE

    patch: Dict[str, Any] = {
        "last_rom_status": rom_pred,
        "last_rom_comparison_status": cmp_status,
        "last_rom_comparison_path": comparison.get("last_rom_comparison_path"),
        "last_rom_median_relative_error": comparison.get("median_relative_error"),
        "last_rom_mean_relative_error": comparison.get("mean_relative_error"),
        "last_rom_p90_relative_error": comparison.get("p90_relative_error"),
        "last_rom_mean_abs_error_hz": comparison.get("mean_abs_error_hz"),
        "last_rom_median_abs_error_hz": comparison.get("median_abs_error_hz"),
        "last_rom_max_abs_error_hz": comparison.get("max_abs_error_hz"),
        "last_rom_matched_mode_count": comparison.get("matched_mode_count"),
        "last_rom_meets_accuracy_target": (comparison.get("accuracy_spec") or {}).get("meets_target"),
        "last_rom_training_sample_count": comparison.get("training_sample_count_at_prediction"),
        "last_rom_error": comparison.get("last_rom_error"),
    }
    return patch


def maybe_run_rom_prepredict(
    *,
    repo_root: Path,
    run_root: Path,
    context: Mapping[str, Any],
    nev: int = DEFAULT_ROM_NEV,
    nonblocking: bool = True,
) -> Dict[str, Any]:
    """Run ROM before FOM; never raises when nonblocking=True."""
    try:
        prediction = run_rom_online_prediction(
            repo_root=repo_root,
            shape_name=str(context["shape_name"]),
            parameters=context["parameters"],
            nev=nev,
        )
        doc = build_rom_prediction_document(context=context, prediction=prediction)
        path = write_rom_prediction_pre_fom(run_root, doc)
        doc["path"] = rel(path, repo_root=repo_root)
        return doc
    except Exception as exc:
        if not nonblocking:
            raise
        doc = build_rom_prediction_document(
            context=context,
            prediction={
                "status": ROM_STATUS_FAILED,
                "method": "ROMManager.solve_online",
                "source": None,
                "confidence": None,
                "runtime_s": 0.0,
                "nev_requested": int(nev),
                "nev_returned": 0,
                "num_basis_modes": 0,
                "frequencies_hz": [],
                "modes": [],
                "error": str(exc),
            },
        )
        try:
            path = write_rom_prediction_pre_fom(run_root, doc)
            doc["path"] = rel(path, repo_root=repo_root)
        except OSError:
            doc["path"] = None
        return doc


def maybe_run_rom_compare(
    *,
    repo_root: Path,
    run_root: Path,
    context: Mapping[str, Any],
    nev: int = DEFAULT_ROM_NEV,
    max_match_distance_hz: Optional[float] = DEFAULT_MAX_MATCH_DISTANCE_HZ,
    nonblocking: bool = True,
    copy_to_project: bool = True,
    write_csv: bool = False,
    rerun_rom_if_missing: bool = True,
) -> Dict[str, Any]:
    """Compare ROM vs M4 FOM catalog; never raises when nonblocking=True."""
    result: Dict[str, Any] = {
        "comparison": None,
        "paths": {},
        "lhs_patch": None,
        "error": None,
    }
    try:
        catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
        fom_modes = load_fom_modes_catalog(catalog_path)
        fom_summary = read_run_production_summary(run_root)
        if str(fom_summary.get("aggregation_status") or "") != AGG_PASS:
            raise RuntimeError(
                f"FOM aggregation not usable: {fom_summary.get('aggregation_status')}"
            )

        rom_doc = load_rom_prediction_pre_fom(run_root)
        if rom_doc is None or str(rom_doc.get("status")) != ROM_STATUS_COMPLETED:
            if rerun_rom_if_missing:
                rom_doc = maybe_run_rom_prepredict(
                    repo_root=repo_root,
                    run_root=run_root,
                    context=context,
                    nev=nev,
                    nonblocking=False,
                )
            elif rom_doc is None:
                raise FileNotFoundError(f"ROM pre-prediction missing: {rom_prediction_path(run_root)}")

        comparison = build_rom_fom_comparison(
            context=context,
            rom_prediction=rom_doc,
            fom_summary=fom_summary,
            fom_modes=fom_modes,
            max_match_distance_hz=max_match_distance_hz,
            rom_prediction_path_rel=rel(rom_prediction_path(run_root), repo_root=repo_root)
            if rom_prediction_path(run_root).is_file()
            else None,
            fom_catalog_path_rel=rel(catalog_path, repo_root=repo_root),
        )
        paths = write_rom_fom_comparison_artifacts(
            repo_root=repo_root,
            run_root=run_root,
            comparison=comparison,
            copy_to_project=copy_to_project,
            write_csv=write_csv,
        )
        comparison["last_rom_comparison_path"] = rel(paths["run_comparison"], repo_root=repo_root)
        lhs_patch = lhs_pool_rom_patch_from_comparison(comparison)
        lhs_patch["last_rom_comparison_path"] = comparison["last_rom_comparison_path"]
        result["comparison"] = comparison
        result["paths"] = {k: str(v) for k, v in paths.items()}
        result["lhs_patch"] = lhs_patch
        return result
    except Exception as exc:
        if not nonblocking:
            raise
        result["error"] = str(exc)
        fail_comparison = {
            "schema": ROM_FOM_COMPARISON_SCHEMA,
            "generated_utc": utc_now(),
            "sample_id": context.get("sample_id"),
            "lhs_row_index": context.get("lhs_row_index"),
            "run_id": context.get("run_id"),
            "shape_name": context.get("shape_name"),
            "status": ROM_COMPARE_FAILED,
            "rom_prediction_status": ROM_STATUS_FAILED,
            "last_rom_error": str(exc),
        }
        result["lhs_patch"] = lhs_pool_rom_patch_from_comparison(
            {**fail_comparison, "last_rom_comparison_path": None}
        )
        return result


def sync_lhs_pool_rom_fields(
    pool: Dict[str, Any],
    *,
    sample_id: str,
    lhs_patch: Mapping[str, Any],
    lhs_path: Path,
) -> None:
    sync_lhs_pool_entry(pool, sample_id=sample_id, patch=dict(lhs_patch))
    write_lhs_pool_with_backup(lhs_path, pool)


def select_completed_lhs_for_rom_compare(
    pool: Mapping[str, Any],
    *,
    completed_only: bool = True,
    max_samples: Optional[int] = None,
    force_sample: Optional[str] = None,
    run_id_suffix: str = DEFAULT_RUN_ID_SUFFIX,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, entry in enumerate(pool.get("entries") or []):
        sid = str(entry.get("id") or "").strip()
        if not sid:
            continue
        if force_sample and sid != force_sample:
            continue
        run_id = str(entry.get("last_run_id") or f"{sid}_{run_id_suffix}")
        status = str(entry.get("status") or "").upper()
        agg = str(entry.get("last_aggregation_status") or "")
        if completed_only and not force_sample:
            if status not in ("COMPLETED", "PASS") and agg != AGG_PASS:
                continue
        rows.append(
            {
                "sample_id": sid,
                "lhs_row_index": i,
                "run_id": run_id,
                "entry": entry,
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows
