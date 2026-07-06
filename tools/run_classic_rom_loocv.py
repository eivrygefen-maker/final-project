#!/usr/bin/env python3
"""VM-only Classical M4 ROM leave-one-out validation.

Usage:
  python tools/run_classic_rom_loocv.py --dry-run
  python tools/run_classic_rom_loocv.py --output-dir /tmp/classic_rom_loocv

This tool reads retained ROM/classic metadata and active M4 surrogate arrays only.
It never calls FEM, GMSH, STK, WAV generation, or raw pipeline_runs artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
M4_SCRIPT_DIR = (
    REPO_ROOT
    / "FEM"
    / "experiments"
    / "active_domain_validation"
    / "physics_integrity"
    / "scripts"
)
if str(M4_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(M4_SCRIPT_DIR))

from v2_b3_m4_modal_surrogate_lib import (  # noqa: E402
    build_surrogate_from_training_rows,
    load_surrogate_model,
    predict_modal_catalog,
)
from v2_b3_m4_rom_fom_compare_lib import (  # noqa: E402
    ACCURACY_BAND_HZ,
    DEFAULT_MAX_MATCH_DISTANCE_HZ,
    TARGET_MEDIAN_RELATIVE_ERROR,
    filter_fom_modes_to_band,
    filter_rom_frequencies_to_band,
    greedy_nearest_hz_match,
)


SHAPE_NAME = "classic"
OFFICIAL_DATASET = REPO_ROOT / "ROM" / SHAPE_NAME / "official_rom_dataset.jsonl"
LHS_POOL = REPO_ROOT / "ROM" / SHAPE_NAME / "lhs_pool.json"
SURROGATE_JSON = REPO_ROOT / "ROM" / SHAPE_NAME / "m4_modal_surrogate.json"
SURROGATE_NPZ = REPO_ROOT / "ROM" / SHAPE_NAME / "m4_modal_surrogate.npz"
SUMMARY_NAME = "loocv_summary.json"
CSV_NAME = "loocv_per_sample.csv"
PNG_NAME = "loocv_modal_error.png"


class LoocvDataError(RuntimeError):
    """Raised when retained ROM data cannot support a valid LOOCV run."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.gettempdir()) / f"classic_rom_loocv_{stamp}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise LoocvDataError(f"expected JSON object: {path}")
    return data


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise LoocvDataError(f"missing retained official ROM dataset: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError as exc:
                raise LoocvDataError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(rec, dict):
                raise LoocvDataError(f"expected JSON object at {path}:{line_no}")
            rows.append(rec)
    if not rows:
        raise LoocvDataError(f"official ROM dataset is empty: {path}")
    return rows


def lhs_parameters_by_sample(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    pool = load_json(path)
    params: Dict[str, Dict[str, Any]] = {}
    indices: Dict[str, int] = {}
    for idx, entry in enumerate(pool.get("entries") or []):
        if not isinstance(entry, dict):
            continue
        sample_id = str(entry.get("id") or entry.get("sample_id") or "").strip()
        if not sample_id:
            continue
        raw_params = entry.get("parameters")
        if isinstance(raw_params, dict):
            params[sample_id] = dict(raw_params)
            indices[sample_id] = int(idx)
    if not params:
        raise LoocvDataError(f"no sample parameters found in retained LHS pool: {path}")
    return params, indices


def mode_frequencies_from_model(model: Mapping[str, Any], row_index: int) -> List[float]:
    arrays = model.get("arrays") or {}
    frequencies = arrays.get("frequencies")
    mode_counts = arrays.get("mode_counts")
    if frequencies is None or mode_counts is None:
        raise LoocvDataError(
            "active M4 surrogate arrays do not contain retained modal targets "
            f"({SURROGATE_NPZ})"
        )
    count = int(mode_counts[row_index])
    freqs = [float(v) for v in frequencies[row_index, :count] if math.isfinite(float(v))]
    freqs.sort()
    return freqs


def build_retained_training_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    registry_rows = load_jsonl(OFFICIAL_DATASET)
    params_by_sample, lhs_indices = lhs_parameters_by_sample(LHS_POOL)
    model = load_surrogate_model(REPO_ROOT, SHAPE_NAME)
    samples = list(model.get("training_samples") or [])
    if not samples:
        raise LoocvDataError(f"active M4 surrogate has no training_samples: {SURROGATE_JSON}")

    registry_ids = {str(r.get("sample_id") or "").strip() for r in registry_rows}
    registry_ids.discard("")
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for row_index, sample_meta in enumerate(samples):
        sample_id = str(sample_meta.get("sample_id") or "").strip()
        if not sample_id:
            skipped.append({"row_index": row_index, "reason": "missing_sample_id"})
            continue
        params = params_by_sample.get(sample_id)
        if not params:
            skipped.append({"sample_id": sample_id, "reason": "missing_lhs_parameters"})
            continue
        freqs = mode_frequencies_from_model(model, row_index)
        modes = [{"frequency_hz": f} for f in freqs]
        band_modes = filter_fom_modes_to_band(modes, band=ACCURACY_BAND_HZ)
        if not band_modes:
            skipped.append({"sample_id": sample_id, "reason": "no_modes_in_validation_band"})
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "lhs_row_index": int(sample_meta.get("lhs_row_index") or lhs_indices.get(sample_id, row_index)),
                "run_id": str(sample_meta.get("run_id") or ""),
                "shape_name": SHAPE_NAME,
                "parameters": dict(params),
                "frequencies_hz": freqs,
                "mode_catalog": modes,
                "mode_count": len(freqs),
                "raw_mode_count": sample_meta.get("raw_mode_count") or len(freqs),
                "deduped_mode_count": sample_meta.get("deduped_mode_count") or len(freqs),
                "catalog_path": None,
                "retained_modal_source": str(SURROGATE_NPZ.relative_to(REPO_ROOT)),
                "listed_in_official_dataset_registry": sample_id in registry_ids,
            }
        )

    if len(rows) < 3:
        raise LoocvDataError(
            "retained Classical ROM data is insufficient for LOOCV: "
            f"need at least 3 valid samples, found {len(rows)}. "
            "The tool did not read deleted/raw pipeline_runs artifacts."
        )

    metadata = {
        "official_dataset_path": str(OFFICIAL_DATASET.relative_to(REPO_ROOT)),
        "official_dataset_registry_entry_count": len(registry_rows),
        "official_dataset_inline_modal_targets": any(
            isinstance(r.get("frequencies_hz"), list) or isinstance(r.get("mode_catalog"), list)
            for r in registry_rows
        ),
        "active_surrogate_json": str(SURROGATE_JSON.relative_to(REPO_ROOT)),
        "active_surrogate_npz": str(SURROGATE_NPZ.relative_to(REPO_ROOT)),
        "active_surrogate_training_sample_count": len(samples),
        "valid_loocv_sample_count": len(rows),
        "skipped": skipped,
    }
    return rows, metadata


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def run_one_fold(
    *,
    holdout: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    k_neighbors: int,
) -> Dict[str, Any]:
    model = build_surrogate_from_training_rows(
        shape_name=SHAPE_NAME,
        training_rows=training_rows,
        k_neighbors=k_neighbors,
    )
    model["holdout_validation"] = True
    model["excluded_sample_ids"] = [holdout["sample_id"]]

    pred = predict_modal_catalog(
        model,
        holdout["parameters"],
        nev=len(holdout["frequencies_hz"]),
    )
    rom_freqs = filter_rom_frequencies_to_band(pred.get("frequencies_hz") or [], band=ACCURACY_BAND_HZ)
    fom_modes = filter_fom_modes_to_band(holdout.get("mode_catalog") or [], band=ACCURACY_BAND_HZ)
    matches, match_meta = greedy_nearest_hz_match(
        rom_frequencies_hz=rom_freqs,
        fom_modes=fom_modes,
        max_match_distance_hz=DEFAULT_MAX_MATCH_DISTANCE_HZ,
    )
    rel_errors = [
        float(m["relative_error"])
        for m in matches
        if m.get("relative_error") is not None and math.isfinite(float(m["relative_error"]))
    ]
    median_rel = statistics.median(rel_errors) if rel_errors else None
    mean_rel = statistics.mean(rel_errors) if rel_errors else None
    p90_rel = percentile(rel_errors, 90.0)
    max_rel = max(rel_errors) if rel_errors else None
    meets = bool(median_rel is not None and median_rel <= TARGET_MEDIAN_RELATIVE_ERROR)
    return {
        "sample_id": holdout["sample_id"],
        "run_id": holdout.get("run_id"),
        "lhs_row_index": holdout.get("lhs_row_index"),
        "training_sample_count": len(training_rows),
        "rom_mode_count_in_band": match_meta["rom_mode_count"],
        "fom_mode_count_in_band": match_meta["fom_mode_count"],
        "matched_mode_count": match_meta["matched_mode_count"],
        "unmatched_rom_count": match_meta["unmatched_rom_count"],
        "unmatched_fom_count": match_meta["unmatched_fom_count"],
        "median_relative_error": median_rel,
        "mean_relative_error": mean_rel,
        "p90_relative_error": p90_rel,
        "max_relative_error": max_rel,
        "median_relative_error_percent": None if median_rel is None else median_rel * 100.0,
        "meets_5_percent_target": meets,
        "neighbor_sample_ids": pred.get("neighbor_sample_ids") or [],
        "matching_method": match_meta["method"],
        "max_match_distance_hz": match_meta["max_match_distance_hz"],
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "sample_id",
        "run_id",
        "lhs_row_index",
        "training_sample_count",
        "rom_mode_count_in_band",
        "fom_mode_count_in_band",
        "matched_mode_count",
        "unmatched_rom_count",
        "unmatched_fom_count",
        "median_relative_error",
        "median_relative_error_percent",
        "mean_relative_error",
        "p90_relative_error",
        "max_relative_error",
        "meets_5_percent_target",
        "neighbor_sample_ids",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["neighbor_sample_ids"] = ",".join(str(v) for v in out.get("neighbor_sample_ids") or [])
            writer.writerow({field: out.get(field) for field in fields})


def write_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    labels = [str(r["sample_id"]) for r in rows]
    values = [
        float(r["median_relative_error_percent"])
        if r.get("median_relative_error_percent") is not None
        else float("nan")
        for r in rows
    ]
    fig_width = max(12.0, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    ax.bar(range(len(labels)), values, color="#4c78a8", width=0.8)
    ax.axhline(TARGET_MEDIAN_RELATIVE_ERROR * 100.0, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_title("Classical ROM LOOCV median modal-frequency error")
    ax.set_xlabel("Held-out sample")
    ax.set_ylabel("Median relative modal-frequency error (%)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_summary(
    *,
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    medians = [
        float(r["median_relative_error"])
        for r in rows
        if r.get("median_relative_error") is not None and math.isfinite(float(r["median_relative_error"]))
    ]
    meeting = sum(1 for r in rows if r.get("meets_5_percent_target"))
    return {
        "schema": "classic_rom_loocv_summary_v1",
        "generated_utc": utc_now_iso(),
        "shape_name": SHAPE_NAME,
        "sample_count": len(rows),
        "fold_count": len(rows),
        "frequency_band_hz": list(ACCURACY_BAND_HZ),
        "target_median_relative_error": TARGET_MEDIAN_RELATIVE_ERROR,
        "target_median_relative_error_percent": TARGET_MEDIAN_RELATIVE_ERROR * 100.0,
        "global_median_error": None if not medians else statistics.median(medians),
        "global_median_error_percent": None if not medians else statistics.median(medians) * 100.0,
        "mean_error": None if not medians else statistics.mean(medians),
        "mean_error_percent": None if not medians else statistics.mean(medians) * 100.0,
        "p90_error": percentile(medians, 90.0),
        "p90_error_percent": None if percentile(medians, 90.0) is None else percentile(medians, 90.0) * 100.0,
        "max_error": None if not medians else max(medians),
        "max_error_percent": None if not medians else max(medians) * 100.0,
        "count_meeting_5_percent_target": meeting,
        "fraction_meeting_5_percent_target": None if not rows else meeting / len(rows),
        "matching_method": "greedy_nearest_hz_one_to_one",
        "max_match_distance_hz": DEFAULT_MAX_MATCH_DISTANCE_HZ,
        "modal_target_source": metadata.get("active_surrogate_npz"),
        "official_dataset_registry": {
            "path": metadata.get("official_dataset_path"),
            "entry_count": metadata.get("official_dataset_registry_entry_count"),
            "contains_inline_modal_targets": metadata.get("official_dataset_inline_modal_targets"),
        },
        "output_files": {
            "summary_json": str((output_dir / SUMMARY_NAME).resolve()),
            "per_sample_csv": str((output_dir / CSV_NAME).resolve()),
            "modal_error_png": str((output_dir / PNG_NAME).resolve()),
        },
        "data_availability": dict(metadata),
    }


def is_inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
        return True
    except ValueError:
        return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VM-only Classical M4 ROM leave-one-out cross-validation."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate retained data and print fold count only.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Output directory for loocv_summary.json, loocv_per_sample.csv, and loocv_modal_error.png.",
    )
    parser.add_argument(
        "--allow-repo-output",
        action="store_true",
        help="Allow --output-dir inside the repository. Default is to refuse repo-local output.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="KNN neighbor count for each in-memory held-out surrogate.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if int(args.k_neighbors) < 1:
        raise LoocvDataError("--k-neighbors must be >= 1")

    rows, metadata = build_retained_training_rows()
    fold_count = len(rows)
    print(f"CLASSIC_ROM_LOOCV_DATA_READY samples={fold_count} band_hz={ACCURACY_BAND_HZ[0]}-{ACCURACY_BAND_HZ[1]}")
    print(
        "CLASSIC_ROM_LOOCV_SOURCES "
        f"registry_entries={metadata['official_dataset_registry_entry_count']} "
        f"surrogate_samples={metadata['active_surrogate_training_sample_count']} "
        f"inline_registry_targets={metadata['official_dataset_inline_modal_targets']}"
    )
    if args.dry_run:
        print(f"CLASSIC_ROM_LOOCV_DRY_RUN folds={fold_count}")
        return 0

    out_dir = args.output_dir
    if is_inside_repo(out_dir) and not args.allow_repo_output:
        raise LoocvDataError(
            f"refusing to write LOOCV outputs inside repository: {out_dir}. "
            "Use an outside path such as /tmp/classic_rom_loocv or pass --allow-repo-output."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for idx, holdout in enumerate(rows, start=1):
        train = [r for r in rows if r["sample_id"] != holdout["sample_id"]]
        print(f"CLASSIC_ROM_LOOCV_FOLD {idx}/{fold_count} holdout={holdout['sample_id']} train={len(train)}")
        results.append(run_one_fold(holdout=holdout, training_rows=train, k_neighbors=int(args.k_neighbors)))

    summary = build_summary(rows=results, metadata=metadata, output_dir=out_dir)
    summary_path = out_dir / SUMMARY_NAME
    csv_path = out_dir / CSV_NAME
    png_path = out_dir / PNG_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(csv_path, results)
    write_plot(png_path, results)
    print(f"CLASSIC_ROM_LOOCV_DONE summary={summary_path} csv={csv_path} png={png_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LoocvDataError as exc:
        print(f"CLASSIC_ROM_LOOCV_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
