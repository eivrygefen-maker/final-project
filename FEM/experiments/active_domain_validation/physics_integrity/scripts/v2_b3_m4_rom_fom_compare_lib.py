#!/usr/bin/env python3
"""M4 ROM pre-prediction and ROM/FOM frequency comparison (no legacy FOM rerun)."""
from __future__ import annotations

import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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

ROM_PREDICTION_SCHEMA_V1 = "rom_prediction_pre_fom_v1"
ROM_PREDICTION_SCHEMA_V2 = "m4_rom_prediction_v2"
ROM_PREDICTION_SCHEMA = ROM_PREDICTION_SCHEMA_V2
ROM_FOM_COMPARISON_SCHEMA = "rom_fom_comparison_v3"
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

VALIDATION_TRAIN_INCLUDED = "train_included"
VALIDATION_HOLDOUT = "holdout"
VALIDATION_LEAVE_ONE_OUT = "leave_one_out"

ROM_STATUS_COMPLETED = "COMPLETED"
ROM_STATUS_FAILED = "FAILED"
ROM_STATUS_SKIPPED = "SKIPPED"


def _median(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.median(vals), 8) if vals else None


def _mae(vals: Sequence[float]) -> Optional[float]:
    return round(statistics.mean(vals), 8) if vals else None

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


@dataclass(frozen=True)
class SurrogateHelpers:
    load_surrogate_model: Callable[..., Any]
    predict_modal_catalog: Callable[..., Any]
    predict_modal_frequencies: Callable[..., Any]
    resolve_active_rom_backend: Callable[..., Any]
    surrogate_json_path: Callable[..., Any]
    build_holdout_surrogate_model: Callable[..., Any]
    production_surrogate_training_sample_ids: Callable[..., Any]


_SURROGATE_HELPERS: Optional[SurrogateHelpers] = None


def get_surrogate_helpers() -> SurrogateHelpers:
    """Lazy import/cache of M4 surrogate helpers (named fields, no tuple unpacking)."""
    global _SURROGATE_HELPERS
    if _SURROGATE_HELPERS is None:
        from v2_b3_m4_modal_surrogate_lib import (  # noqa: WPS433
            build_holdout_surrogate_model,
            load_surrogate_model,
            predict_modal_catalog,
            predict_modal_frequencies,
            production_surrogate_training_sample_ids,
            resolve_active_rom_backend,
            surrogate_json_path,
        )

        _SURROGATE_HELPERS = SurrogateHelpers(
            load_surrogate_model=load_surrogate_model,
            predict_modal_catalog=predict_modal_catalog,
            predict_modal_frequencies=predict_modal_frequencies,
            resolve_active_rom_backend=resolve_active_rom_backend,
            surrogate_json_path=surrogate_json_path,
            build_holdout_surrogate_model=build_holdout_surrogate_model,
            production_surrogate_training_sample_ids=production_surrogate_training_sample_ids,
        )
    return _SURROGATE_HELPERS


def resolve_validation_metadata(
    *,
    target_sample_id: str,
    training_sample_ids: Sequence[str],
    excluded_sample_ids: Sequence[str],
    validation_mode: str,
) -> Dict[str, Any]:
    train_ids = [str(s) for s in training_sample_ids if str(s)]
    excluded = [str(s) for s in excluded_sample_ids if str(s)]
    includes_target = str(target_sample_id) in train_ids
    if validation_mode in (VALIDATION_HOLDOUT, VALIDATION_LEAVE_ONE_OUT):
        includes_target = False
    meaningful = not includes_target and validation_mode != VALIDATION_TRAIN_INCLUDED
    return {
        "validation_mode": validation_mode,
        "training_includes_target": includes_target,
        "training_sample_ids": train_ids,
        "excluded_sample_ids": excluded,
        "training_sample_count_at_prediction": len(train_ids),
        "accuracy_meaningful": meaningful,
    }


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


def _nearest_mode_by_frequency(
    modes: Sequence[Mapping[str, Any]],
    target_hz: float,
) -> Dict[str, Any]:
    best: Dict[str, Any] = {}
    best_d = float("inf")
    for m in modes:
        try:
            f = float(m.get("frequency_hz"))
        except (TypeError, ValueError):
            continue
        d = abs(f - target_hz)
        if d < best_d:
            best_d = d
            best = dict(m)
    return best


def _enrich_matches_with_phase2_scalars(
    matches: List[Dict[str, Any]],
    *,
    rom_modes: Sequence[Mapping[str, Any]],
    fom_modes: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    from v2_b3_m4_rom_scalar_fields import enrich_match_with_phase2_fields  # noqa: WPS433

    enriched: List[Dict[str, Any]] = []
    for m in matches:
        rom_hz = float(m.get("rom_frequency_hz") or 0.0)
        fom_hz = float(m.get("fom_frequency_hz") or 0.0)
        rom_mode = _nearest_mode_by_frequency(rom_modes, rom_hz)
        fom_mode = _nearest_mode_by_frequency(fom_modes, fom_hz)
        enriched.append(enrich_match_with_phase2_fields(m, rom_mode=rom_mode, fom_mode=fom_mode))
    return enriched


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


def _accuracy_spec_block(
    metrics: Mapping[str, Any],
    *,
    accuracy_meaningful: bool,
) -> Dict[str, Any]:
    median_rel = metrics.get("median_relative_error")
    meets_raw = (
        median_rel is not None
        and median_rel == median_rel
        and float(median_rel) <= TARGET_MEDIAN_RELATIVE_ERROR
    )
    meets_meaningful = bool(meets_raw and accuracy_meaningful)
    return {
        "frequency_band_hz": list(ACCURACY_BAND_HZ),
        "primary_metric": PRIMARY_ACCURACY_METRIC,
        "target_median_relative_error": TARGET_MEDIAN_RELATIVE_ERROR,
        "accuracy_meaningful": bool(accuracy_meaningful),
        "meets_target": bool(meets_raw) if median_rel is not None else False,
        "meets_target_meaningful": meets_meaningful if median_rel is not None else False,
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
    surrogate_model: Optional[Mapping[str, Any]] = None,
    validation_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    helpers = get_surrogate_helpers()
    t0 = time.perf_counter()
    model = (
        dict(surrogate_model)
        if surrogate_model is not None
        else helpers.load_surrogate_model(repo_root, shape_name)
    )
    out = helpers.predict_modal_catalog(model, parameters, nev=int(nev))
    elapsed = time.perf_counter() - t0
    freqs = [float(f) for f in (out.get("frequencies_hz") or [])]
    predicted_modes = list(out.get("predicted_modes") or [])
    modes = predicted_modes if predicted_modes else [_unavailable_mode_record(f) for f in freqs]
    holdout = bool(model.get("holdout_validation"))
    source = (
        f"holdout:{','.join(model.get('excluded_sample_ids') or [])}"
        if holdout
        else str(helpers.surrogate_json_path(repo_root, shape_name))
    )
    train_ids = list((validation_meta or {}).get("training_sample_ids") or [])
    if not train_ids:
        train_ids = [str(s.get("sample_id") or "") for s in (model.get("training_samples") or []) if s.get("sample_id")]
    return {
        "status": ROM_STATUS_COMPLETED,
        "method": str(out.get("method") or "m4_modal_surrogate"),
        "source": source,
        "confidence": "m4_fom_knn_surrogate_holdout" if holdout else "m4_fom_knn_surrogate",
        "runtime_s": round(elapsed, 4),
        "nev_requested": int(nev),
        "nev_returned": int(out.get("nev_returned") or len(freqs)),
        "num_basis_modes": int(model.get("training_sample_count") or len(train_ids)),
        "training_sample_ids": train_ids,
        "frequencies_hz": [round(f, 6) for f in freqs],
        "predicted_modes": modes,
        "modes": modes,
        "rom_prediction_runtime_s": round(elapsed, 4),
        "error": None,
        "raw": out,
        "validation_meta": dict(validation_meta) if validation_meta else None,
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
    surrogate_model: Optional[Mapping[str, Any]] = None,
    validation_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Predict modal frequencies using the best available ROM backend:
    1) M4 modal surrogate trained from modes_catalog.jsonl (preferred)
    2) Legacy POD reduced_basis.npz + operator projection (optional fallback)
    """
    helpers = get_surrogate_helpers()
    backend = (
        "m4_surrogate"
        if surrogate_model is not None
        else helpers.resolve_active_rom_backend(repo_root, shape_name)
    )
    t0 = time.perf_counter()
    try:
        if backend == "m4_surrogate":
            return _run_m4_surrogate_prediction(
                repo_root=repo_root,
                shape_name=shape_name,
                parameters=parameters,
                nev=nev,
                surrogate_model=surrogate_model,
                validation_meta=validation_meta,
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
    predicted_modes = list(prediction.get("predicted_modes") or prediction.get("modes") or [])
    has_scalars = any(
        m.get("top_share") is not None or m.get("radiation_proxy") is not None for m in predicted_modes
    )
    schema = ROM_PREDICTION_SCHEMA_V2 if has_scalars else ROM_PREDICTION_SCHEMA_V1
    runtime_s = prediction.get("rom_prediction_runtime_s") or prediction.get("runtime_s")
    return {
        "schema": schema,
        "generated_utc": utc_now(),
        "sample_id": context["sample_id"],
        "lhs_row_index": context["lhs_row_index"],
        "shape_name": context["shape_name"],
        "parameters": dict(context.get("parameters") or {}),
        "run_id": context.get("run_id"),
        "status": prediction.get("status"),
        "prediction_method": prediction.get("method"),
        "method": prediction.get("method"),
        "source": prediction.get("source"),
        "confidence": prediction.get("confidence"),
        "runtime_s": runtime_s,
        "rom_prediction_runtime_s": runtime_s,
        "nev_requested": prediction.get("nev_requested"),
        "nev_returned": prediction.get("nev_returned"),
        "num_basis_modes": prediction.get("num_basis_modes"),
        "training_sample_count": _training_sample_count_from_prediction(prediction),
        "training_sample_count_at_prediction": _training_sample_count_from_prediction(prediction),
        "frequencies_hz": list(prediction.get("frequencies_hz") or []),
        "predicted_modes": predicted_modes,
        "modes": predicted_modes,
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
    validation_meta: Optional[Mapping[str, Any]] = None,
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
    rom_predicted_modes = list(rom_prediction.get("predicted_modes") or rom_prediction.get("modes") or [])
    if rom_predicted_modes:
        matches = _enrich_matches_with_phase2_scalars(
            matches,
            rom_modes=rom_predicted_modes,
            fom_modes=fom_modes_band,
        )
    metrics = _error_metrics(matches)
    from v2_b3_m4_rom_scalar_fields import compute_phase2_scalar_metrics  # noqa: WPS433

    phase2_block = compute_phase2_scalar_metrics(matches) if rom_predicted_modes else {}
    vmeta = dict(validation_meta or rom_prediction.get("validation_meta") or {})
    if not vmeta:
        train_ids = list(rom_prediction.get("training_sample_ids") or [])
        target_sid = str(context["sample_id"])
        includes = target_sid in train_ids
        vmeta = resolve_validation_metadata(
            target_sample_id=target_sid,
            training_sample_ids=train_ids,
            excluded_sample_ids=[],
            validation_mode=VALIDATION_TRAIN_INCLUDED if includes else VALIDATION_HOLDOUT,
        )
    training_count = int(
        vmeta.get("training_sample_count_at_prediction")
        or _training_sample_count_from_prediction(rom_prediction)
        or 0
    )
    accuracy_meaningful = bool(vmeta.get("accuracy_meaningful"))
    accuracy_spec = _accuracy_spec_block(metrics, accuracy_meaningful=accuracy_meaningful)
    warnings: List[str] = []
    if vmeta.get("training_includes_target"):
        warnings.append(
            "train_test_leakage: target sample included in surrogate training set; "
            "accuracy metrics are not meaningful"
        )

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
        "validation_mode": vmeta.get("validation_mode"),
        "training_includes_target": vmeta.get("training_includes_target"),
        "training_sample_ids": list(vmeta.get("training_sample_ids") or []),
        "excluded_sample_ids": list(vmeta.get("excluded_sample_ids") or []),
        "training_sample_count_at_prediction": training_count,
        "accuracy_meaningful": accuracy_meaningful,
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
        **phase2_block,
        "per_mode_matches": matches,
        "rom_prediction_runtime_s": rom_prediction.get("rom_prediction_runtime_s")
        or rom_prediction.get("runtime_s"),
        "rom_comparison_runtime_s": None,
        "total_rom_runtime_s": None,
        "rom_frequencies_hz": rom_freqs,
        "rom_frequencies_hz_total": [round(float(f), 6) for f in rom_freqs_all],
        "fom_frequencies_hz": [round(float(r["frequency_hz"]), 6) for r in fom_modes_band],
        "fom_frequencies_hz_total": [round(float(r["frequency_hz"]), 6) for r in fom_modes_all],
        "status": compare_status,
        "warnings": warnings,
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
            "validation_mode": comparison.get("validation_mode"),
            "training_includes_target": comparison.get("training_includes_target"),
            "accuracy_meaningful": comparison.get("accuracy_meaningful"),
            "meets_target": (comparison.get("accuracy_spec") or {}).get("meets_target"),
            "meets_target_meaningful": (comparison.get("accuracy_spec") or {}).get(
                "meets_target_meaningful"
            ),
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
    "validation_mode",
    "training_includes_target",
    "accuracy_meaningful",
    "excluded_sample_ids",
    "training_sample_count_at_prediction",
    "rom_prediction_method",
    "matched_mode_count",
    "median_relative_error",
    "mean_relative_error",
    "p90_relative_error",
    "median_abs_error_hz",
    "mean_abs_error_hz",
    "top_share_mae",
    "back_share_mae",
    "air_share_mae",
    "radiation_proxy_relative_error_median",
    "mic_output_proxy_relative_error_median",
    "coupling_class_accuracy",
    "dominant_region_accuracy",
    "audio_weighted_frequency_error",
    "audio_weighted_output_proxy_error",
    "rom_prediction_runtime_s",
    "rom_comparison_runtime_s",
    "total_rom_runtime_s",
    "meets_target",
    "meets_target_meaningful",
    "status",
)


def _comparison_to_history_row(comparison: Mapping[str, Any]) -> Dict[str, Any]:
    spec = comparison.get("accuracy_spec") or {}
    excluded = comparison.get("excluded_sample_ids") or []
    return {
        "recorded_utc": comparison.get("generated_utc") or utc_now(),
        "sample_id": comparison.get("sample_id"),
        "run_id": comparison.get("run_id"),
        "validation_mode": comparison.get("validation_mode"),
        "training_includes_target": comparison.get("training_includes_target"),
        "accuracy_meaningful": comparison.get("accuracy_meaningful"),
        "excluded_sample_ids": ";".join(str(s) for s in excluded if s),
        "training_sample_count_at_prediction": comparison.get("training_sample_count_at_prediction"),
        "rom_prediction_method": comparison.get("rom_prediction_method"),
        "matched_mode_count": comparison.get("matched_mode_count"),
        "median_relative_error": comparison.get("median_relative_error"),
        "mean_relative_error": comparison.get("mean_relative_error"),
        "p90_relative_error": comparison.get("p90_relative_error"),
        "median_abs_error_hz": comparison.get("median_abs_error_hz"),
        "mean_abs_error_hz": comparison.get("mean_abs_error_hz"),
        "top_share_mae": (comparison.get("phase2_scalar_metrics") or {}).get("top_share_mae"),
        "back_share_mae": (comparison.get("phase2_scalar_metrics") or {}).get("back_share_mae"),
        "air_share_mae": (comparison.get("phase2_scalar_metrics") or {}).get("air_share_mae"),
        "radiation_proxy_relative_error_median": (
            (comparison.get("phase2_scalar_metrics") or {}).get("radiation_proxy_relative_error_median")
        ),
        "mic_output_proxy_relative_error_median": (
            (comparison.get("phase2_scalar_metrics") or {}).get("mic_output_proxy_relative_error_median")
        ),
        "coupling_class_accuracy": (comparison.get("phase2_scalar_metrics") or {}).get(
            "coupling_class_accuracy"
        ),
        "dominant_region_accuracy": (comparison.get("phase2_scalar_metrics") or {}).get(
            "dominant_region_accuracy"
        ),
        "audio_weighted_frequency_error": (comparison.get("phase2_scalar_metrics") or {}).get(
            "audio_weighted_frequency_error"
        ),
        "audio_weighted_output_proxy_error": (comparison.get("phase2_scalar_metrics") or {}).get(
            "audio_weighted_output_proxy_error"
        ),
        "rom_prediction_runtime_s": comparison.get("rom_prediction_runtime_s"),
        "rom_comparison_runtime_s": comparison.get("rom_comparison_runtime_s"),
        "total_rom_runtime_s": comparison.get("total_rom_runtime_s"),
        "meets_target": spec.get("meets_target"),
        "meets_target_meaningful": spec.get("meets_target_meaningful"),
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
            if key == "training_sample_count_at_prediction" or key == "matched_mode_count":
                out[key] = int(float(val))
            else:
                out[key] = float(val)
        except (TypeError, ValueError):
            out[key] = None
    for key in ("meets_target", "meets_target_meaningful", "training_includes_target", "accuracy_meaningful"):
        val = out.get(key)
        if isinstance(val, str):
            out[key] = val.strip().lower() in ("true", "1", "yes")
    return out


def _aggregate_accuracy_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    medians = [
        float(r["median_relative_error"])
        for r in rows
        if r.get("median_relative_error") is not None and r["median_relative_error"] == r["median_relative_error"]
    ]
    latest = rows[-1] if rows else {}
    return {
        "comparison_count": len(rows),
        "median_of_median_relative_error": round(statistics.median(medians), 8) if medians else None,
        "mean_of_median_relative_error": round(statistics.mean(medians), 8) if medians else None,
        "p90_of_median_relative_error": round(_percentile(medians, 90.0), 8) if medians else None,
        "samples_meeting_target": sum(1 for v in medians if v <= TARGET_MEDIAN_RELATIVE_ERROR),
        "samples_meeting_target_fraction": round(
            sum(1 for v in medians if v <= TARGET_MEDIAN_RELATIVE_ERROR) / len(medians), 4
        )
        if medians
        else None,
        "latest_sample_id": latest.get("sample_id"),
        "latest_run_id": latest.get("run_id"),
        "latest_median_relative_error": latest.get("median_relative_error"),
        "latest_training_sample_count_at_prediction": latest.get("training_sample_count_at_prediction"),
        "latest_meets_target": latest.get("meets_target"),
        "latest_meets_target_meaningful": latest.get("meets_target_meaningful"),
        "latest_validation_mode": latest.get("validation_mode"),
        "latest_accuracy_meaningful": latest.get("accuracy_meaningful"),
    }


def build_accuracy_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    parsed = [_coerce_history_numeric(r) for r in rows if r.get("status") == ROM_COMPARE_COMPLETED]
    meaningful = [r for r in parsed if bool(r.get("accuracy_meaningful"))]
    medians = [
        float(r["median_relative_error"])
        for r in parsed
        if r.get("median_relative_error") is not None and r["median_relative_error"] == r["median_relative_error"]
    ]
    meets = [bool(r.get("meets_target")) for r in parsed if r.get("meets_target") is not None]

    def _stk_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {}
        return {
            "top_share_mae_median": _median(
                [float(r["top_share_mae"]) for r in rows if r.get("top_share_mae") not in (None, "")]
            ),
            "radiation_proxy_rel_error_median": _median(
                [
                    float(r["radiation_proxy_relative_error_median"])
                    for r in rows
                    if r.get("radiation_proxy_relative_error_median") not in (None, "")
                ]
            ),
            "mic_output_proxy_rel_error_median": _median(
                [
                    float(r["mic_output_proxy_relative_error_median"])
                    for r in rows
                    if r.get("mic_output_proxy_relative_error_median") not in (None, "")
                ]
            ),
            "audio_weighted_frequency_error_mean": _mae(
                [
                    float(r["audio_weighted_frequency_error"])
                    for r in rows
                    if r.get("audio_weighted_frequency_error") not in (None, "")
                ]
            ),
        }

    def _class_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        accs = [
            float(r["coupling_class_accuracy"])
            for r in rows
            if r.get("coupling_class_accuracy") not in (None, "")
        ]
        doms = [
            float(r["dominant_region_accuracy"])
            for r in rows
            if r.get("dominant_region_accuracy") not in (None, "")
        ]
        return {
            "coupling_class_accuracy_mean": _mae(accs),
            "dominant_region_accuracy_mean": _mae(doms),
        }

    def _runtime_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        preds = [
            float(r["rom_prediction_runtime_s"])
            for r in rows
            if r.get("rom_prediction_runtime_s") not in (None, "")
        ]
        cmps = [
            float(r["rom_comparison_runtime_s"])
            for r in rows
            if r.get("rom_comparison_runtime_s") not in (None, "")
        ]
        totals = [
            float(r["total_rom_runtime_s"]) for r in rows if r.get("total_rom_runtime_s") not in (None, "")
        ]
        return {
            "rom_prediction_runtime_s_median": _median(preds),
            "rom_comparison_runtime_s_median": _median(cmps),
            "total_rom_runtime_s_median": _median(totals),
        }

    by_training: Dict[int, List[Dict[str, Any]]] = {}
    for r in meaningful:
        tc = r.get("training_sample_count_at_prediction")
        if tc is None:
            continue
        by_training.setdefault(int(tc), []).append(dict(r))

    training_breakdown = []
    for tc in sorted(by_training):
        rows_tc = by_training[tc]
        freq_vals = [
            float(r["median_relative_error"])
            for r in rows_tc
            if r.get("median_relative_error") is not None and r["median_relative_error"] == r["median_relative_error"]
        ]
        training_breakdown.append(
            {
                "training_sample_count_at_prediction": tc,
                "comparison_count": len(rows_tc),
                "frequency_accuracy": {
                    "median_of_median_relative_error": round(statistics.median(freq_vals), 8)
                    if freq_vals
                    else None,
                    "mean_of_median_relative_error": round(statistics.mean(freq_vals), 8) if freq_vals else None,
                    "samples_meeting_target": sum(1 for v in freq_vals if v <= TARGET_MEDIAN_RELATIVE_ERROR),
                },
                "stk_audio_scalar_accuracy": _stk_block(rows_tc),
                "classification_accuracy": _class_block(rows_tc),
                "runtime": _runtime_block(rows_tc),
            }
        )

    leakage = [r for r in parsed if not bool(r.get("accuracy_meaningful"))]

    return {
        "schema": ACCURACY_SUMMARY_SCHEMA,
        "generated_utc": utc_now(),
        "comparison_count": len(parsed),
        "meaningful_comparison_count": len(meaningful),
        "leakage_comparison_count": len(leakage),
        "frequency_band_hz": list(ACCURACY_BAND_HZ),
        "primary_metric": PRIMARY_ACCURACY_METRIC,
        "target_median_relative_error": TARGET_MEDIAN_RELATIVE_ERROR,
        "aggregate_all_comparisons": _aggregate_accuracy_block(parsed),
        "aggregate_meaningful_only": _aggregate_accuracy_block(meaningful),
        "aggregate": _aggregate_accuracy_block(meaningful),
        "frequency_accuracy": {
            "meaningful_only": _aggregate_accuracy_block(meaningful),
            "all_comparisons": _aggregate_accuracy_block(parsed),
        },
        "stk_audio_scalar_accuracy": {
            "meaningful_only": _stk_block(meaningful),
            "all_comparisons": _stk_block(parsed),
        },
        "classification_accuracy": {
            "meaningful_only": _class_block(meaningful),
            "all_comparisons": _class_block(parsed),
        },
        "runtime": {
            "meaningful_only": _runtime_block(meaningful),
            "all_comparisons": _runtime_block(parsed),
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
        "last_rom_meets_accuracy_target": (comparison.get("accuracy_spec") or {}).get(
            "meets_target_meaningful"
        ),
        "last_rom_accuracy_meaningful": comparison.get("accuracy_meaningful"),
        "last_rom_validation_mode": comparison.get("validation_mode"),
        "last_rom_training_sample_count": comparison.get("training_sample_count_at_prediction"),
        "last_rom_prediction_runtime_s": comparison.get("rom_prediction_runtime_s"),
        "last_rom_comparison_runtime_s": comparison.get("rom_comparison_runtime_s"),
        "last_rom_total_runtime_s": comparison.get("total_rom_runtime_s"),
        "last_rom_top_share_mae": (comparison.get("phase2_scalar_metrics") or {}).get("top_share_mae"),
        "last_rom_coupling_class_accuracy": (comparison.get("phase2_scalar_metrics") or {}).get(
            "coupling_class_accuracy"
        ),
        "last_rom_error": comparison.get("last_rom_error"),
    }
    return patch


def _production_validation_meta(
    *,
    repo_root: Path,
    shape_name: str,
    target_sample_id: str,
) -> Dict[str, Any]:
    helpers = get_surrogate_helpers()
    train_ids, _ = helpers.production_surrogate_training_sample_ids(repo_root, shape_name)
    if str(target_sample_id) in train_ids:
        return resolve_validation_metadata(
            target_sample_id=target_sample_id,
            training_sample_ids=train_ids,
            excluded_sample_ids=[],
            validation_mode=VALIDATION_TRAIN_INCLUDED,
        )
    return resolve_validation_metadata(
        target_sample_id=target_sample_id,
        training_sample_ids=train_ids,
        excluded_sample_ids=[],
        validation_mode=VALIDATION_HOLDOUT,
    )


def _prepare_holdout_prediction(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    context: Mapping[str, Any],
    nev: int,
    leave_one_out: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    helpers = get_surrogate_helpers()
    target = str(context["sample_id"])
    shape = str(context["shape_name"])
    excluded = [target]
    model, training_rows = helpers.build_holdout_surrogate_model(
        repo_root=repo_root,
        pool=pool,
        shape_name=shape,
        exclude_sample_ids=excluded,
    )
    train_ids = [str(r["sample_id"]) for r in training_rows]
    mode = VALIDATION_LEAVE_ONE_OUT if leave_one_out else VALIDATION_HOLDOUT
    vmeta = resolve_validation_metadata(
        target_sample_id=target,
        training_sample_ids=train_ids,
        excluded_sample_ids=excluded,
        validation_mode=mode,
    )
    prediction = run_rom_online_prediction(
        repo_root=repo_root,
        shape_name=shape,
        parameters=context["parameters"],
        nev=nev,
        surrogate_model=model,
        validation_meta=vmeta,
    )
    return prediction, vmeta


def maybe_run_rom_prepredict(
    *,
    repo_root: Path,
    run_root: Path,
    context: Mapping[str, Any],
    nev: int = DEFAULT_ROM_NEV,
    nonblocking: bool = True,
    pool: Optional[Mapping[str, Any]] = None,
    exclude_target_from_training: bool = False,
    leave_one_out: bool = False,
) -> Dict[str, Any]:
    """Run ROM before FOM; never raises when nonblocking=True."""
    try:
        if (exclude_target_from_training or leave_one_out) and pool is not None:
            prediction, vmeta = _prepare_holdout_prediction(
                repo_root=repo_root,
                pool=pool,
                context=context,
                nev=nev,
                leave_one_out=leave_one_out,
            )
        else:
            vmeta = _production_validation_meta(
                repo_root=repo_root,
                shape_name=str(context["shape_name"]),
                target_sample_id=str(context["sample_id"]),
            )
            prediction = run_rom_online_prediction(
                repo_root=repo_root,
                shape_name=str(context["shape_name"]),
                parameters=context["parameters"],
                nev=nev,
                validation_meta=vmeta,
            )
        doc = build_rom_prediction_document(context=context, prediction=prediction)
        doc["validation_meta"] = vmeta
        doc["validation_mode"] = vmeta.get("validation_mode")
        doc["training_includes_target"] = vmeta.get("training_includes_target")
        doc["training_sample_ids"] = list(vmeta.get("training_sample_ids") or [])
        doc["excluded_sample_ids"] = list(vmeta.get("excluded_sample_ids") or [])
        doc["accuracy_meaningful"] = vmeta.get("accuracy_meaningful")
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
    pool: Optional[Mapping[str, Any]] = None,
    exclude_target_from_training: bool = False,
    leave_one_out: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Compare ROM vs M4 FOM catalog; never raises when nonblocking=True."""
    result: Dict[str, Any] = {
        "comparison": None,
        "paths": {},
        "lhs_patch": None,
        "error": None,
    }
    t_compare = time.perf_counter()
    try:
        catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
        fom_modes = load_fom_modes_catalog(catalog_path)
        fom_summary = read_run_production_summary(run_root)
        if str(fom_summary.get("aggregation_status") or "") != AGG_PASS:
            raise RuntimeError(
                f"FOM aggregation not usable: {fom_summary.get('aggregation_status')}"
            )

        use_holdout = bool(exclude_target_from_training or leave_one_out)
        validation_meta: Optional[Dict[str, Any]] = None

        if use_holdout:
            if pool is None:
                raise ValueError("pool required for holdout/leave-one-out ROM validation")
            rom_doc = maybe_run_rom_prepredict(
                repo_root=repo_root,
                run_root=run_root,
                context=context,
                nev=nev,
                nonblocking=False,
                pool=pool,
                exclude_target_from_training=exclude_target_from_training,
                leave_one_out=leave_one_out,
            )
            validation_meta = dict(rom_doc.get("validation_meta") or {})
            if not validation_meta:
                validation_meta = resolve_validation_metadata(
                    target_sample_id=str(context["sample_id"]),
                    training_sample_ids=rom_doc.get("training_sample_ids") or [],
                    excluded_sample_ids=rom_doc.get("excluded_sample_ids") or [str(context["sample_id"])],
                    validation_mode=(
                        VALIDATION_LEAVE_ONE_OUT if leave_one_out else VALIDATION_HOLDOUT
                    ),
                )
        else:
            rom_doc = load_rom_prediction_pre_fom(run_root)
            if rom_doc is None or str(rom_doc.get("status")) != ROM_STATUS_COMPLETED:
                if rerun_rom_if_missing:
                    rom_doc = maybe_run_rom_prepredict(
                        repo_root=repo_root,
                        run_root=run_root,
                        context=context,
                        nev=nev,
                        nonblocking=False,
                        pool=pool,
                    )
                elif rom_doc is None:
                    raise FileNotFoundError(
                        f"ROM pre-prediction missing: {rom_prediction_path(run_root)}"
                    )
            validation_meta = _production_validation_meta(
                repo_root=repo_root,
                shape_name=str(context["shape_name"]),
                target_sample_id=str(context["sample_id"]),
            )

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
            validation_meta=validation_meta,
        )
        paths = write_rom_fom_comparison_artifacts(
            repo_root=repo_root,
            run_root=run_root,
            comparison=comparison,
            copy_to_project=copy_to_project,
            write_csv=write_csv,
        )
        cmp_elapsed = time.perf_counter() - t_compare
        pred_runtime = float(
            comparison.get("rom_prediction_runtime_s")
            or (rom_doc or {}).get("rom_prediction_runtime_s")
            or (rom_doc or {}).get("runtime_s")
            or 0.0
        )
        comparison["rom_comparison_runtime_s"] = round(cmp_elapsed, 4)
        comparison["total_rom_runtime_s"] = round(pred_runtime + cmp_elapsed, 4)
        write_json_atomic(paths["run_comparison"], dict(comparison))
        if copy_to_project and "project_comparison" in paths:
            write_json_atomic(paths["project_comparison"], dict(comparison))

        comparison["last_rom_comparison_path"] = rel(paths["run_comparison"], repo_root=repo_root)
        lhs_patch = lhs_pool_rom_patch_from_comparison(comparison)
        lhs_patch["last_rom_comparison_path"] = comparison["last_rom_comparison_path"]
        result["comparison"] = comparison
        result["paths"] = {k: str(v) for k, v in paths.items()}
        result["lhs_patch"] = lhs_patch
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        if not nonblocking:
            raise
        result["error"] = str(exc)
        result["traceback"] = tb
        if debug:
            print(tb, file=sys.stderr, flush=True)
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
