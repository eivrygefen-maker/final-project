#!/usr/bin/env python3
"""Generate or extend the BOX LHS pool for FOM/FEM/ROM (``ROM/box/lhs_pool.json``)."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL_PATH = REPO_ROOT / "ROM" / "box" / "lhs_pool.json"
DEFAULT_COUNT = 40
DEFAULT_SEED = 20260616
DEFAULT_FOM_RUN_ID_SUFFIX = "box_fom_v1"
M4_GUITARS_ROOT = (
    REPO_ROOT
    / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
)

WOOD_IDS: Tuple[str, ...] = ("spruce", "cedar", "mahogany", "rosewood", "maple")

# Conservative BOX geometry bounds — same keys as classic M4 LHS (7D + woods).
DEFAULT_BOX_BOUNDS: Dict[str, Any] = {
    "geometry.length": {"min": 0.40, "max": 0.52},
    "geometry.width": {"min": 0.32, "max": 0.42},
    "geometry.depth": {"min": 0.06, "max": 0.16},
    "geometry.top_thickness": {"min": 0.0025, "max": 0.0035},
    "geometry.hole_radius": {"min": 0.035, "max": 0.048},
    "top_wood_id": list(WOOD_IDS),
    "back_wood_id": list(WOOD_IDS),
}

PENDING_ENTRY_TEMPLATE: Dict[str, Any] = {
    "status": "PENDING",
    "error": None,
    "last_run_id": None,
    "last_batch_id": None,
    "last_started_at": None,
    "last_error": None,
    "last_run_dir": None,
    "last_finished_at": None,
    "last_elapsed_s": None,
    "last_aggregation_status": None,
    "last_deduped_mode_count": None,
    "last_participation_computed_count": None,
    "last_audio_coupling_computed_count": None,
}


def _finalize_lhs_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply FEM wood/thickness finalization (``geometry.back_thickness`` from top)."""
    fem_scripts = REPO_ROOT / "FEM" / "scripts"
    if str(fem_scripts) not in sys.path:
        sys.path.insert(0, str(fem_scripts))
    from wood_library import finalize_lhs_thickness_params  # noqa: WPS433

    return finalize_lhs_thickness_params(params)


def box_sample_id(index: int) -> str:
    return f"box_sample_{int(index):03d}"


def parse_box_sample_index(sample_id: str) -> Optional[int]:
    raw = str(sample_id or "").strip()
    if not raw.startswith("box_sample_"):
        return None
    try:
        return int(raw.split("_")[-1])
    except ValueError:
        return None


def box_fom_run_id(sample_id: str, run_id_suffix: str = DEFAULT_FOM_RUN_ID_SUFFIX) -> str:
    return f"{sample_id}_{run_id_suffix}"


def box_fom_run_root(
    repo_root: Path,
    sample_id: str,
    run_id_suffix: str = DEFAULT_FOM_RUN_ID_SUFFIX,
) -> Path:
    return M4_GUITARS_ROOT / sample_id / "runs" / box_fom_run_id(sample_id, run_id_suffix)


def _latin_hypercube_unit(n_samples: int, n_dim: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    perm = np.stack([rng.permutation(n_samples) for _ in range(n_dim)], axis=0)
    return ((perm + rng.random((n_dim, n_samples))) / n_samples).T


def _lhs_values_for_key(spec: Any, uvals) -> List[Any]:
    import numpy as np

    if isinstance(spec, dict) and "min" in spec and "max" in spec:
        vmin = float(spec["min"])
        vmax = float(spec["max"])
        vals = vmin + (vmax - vmin) * uvals
        return [round(float(v), 6) for v in vals.tolist()]
    options = list(spec)
    if not options:
        raise ValueError("Discrete sweep options cannot be empty.")
    idx = np.floor(uvals * len(options)).astype(int)
    idx = np.clip(idx, 0, len(options) - 1)
    return [options[int(i)] for i in idx]


def _bounds_from_pool(pool: Mapping[str, Any]) -> Dict[str, Any]:
    recorded = pool.get("lhs_bounds")
    if isinstance(recorded, dict) and recorded:
        return {k: deepcopy(v) for k, v in recorded.items()}
    sampling = pool.get("sampling")
    if isinstance(sampling, dict):
        bounds = sampling.get("bounds")
        if isinstance(bounds, dict) and bounds:
            return {k: deepcopy(v) for k, v in bounds.items()}
    return deepcopy(DEFAULT_BOX_BOUNDS)


def _infer_bounds_from_entries(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    bounds = deepcopy(DEFAULT_BOX_BOUNDS)
    numeric_keys = (
        "geometry.length",
        "geometry.width",
        "geometry.depth",
        "geometry.top_thickness",
        "geometry.hole_radius",
        "geometry.back_thickness",
    )
    observed: Dict[str, List[float]] = {k: [] for k in numeric_keys}
    woods_top: set[str] = set()
    woods_back: set[str] = set()
    for entry in entries:
        params = entry.get("parameters") or {}
        for key in numeric_keys:
            val = params.get(key)
            if val is not None:
                observed[key].append(float(val))
        tw = params.get("top_wood_id")
        bw = params.get("back_wood_id")
        if tw:
            woods_top.add(str(tw))
        if bw:
            woods_back.add(str(bw))
    for key in numeric_keys:
        vals = observed[key]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        spec = bounds.get(key)
        if isinstance(spec, dict):
            spec["min"] = min(float(spec["min"]), lo)
            spec["max"] = max(float(spec["max"]), hi)
    if woods_top:
        bounds["top_wood_id"] = sorted(woods_top | set(WOOD_IDS))
    if woods_back:
        bounds["back_wood_id"] = sorted(woods_back | set(WOOD_IDS))
    return bounds


def _build_parameters_row(cols: Mapping[str, Sequence[Any]], index: int) -> Dict[str, Any]:
    params = {k: cols[k][index] for k in sorted(cols.keys())}
    params["geometry.shape_type"] = "Box"
    return _finalize_lhs_params(params)


def _blank_entry(sample_id: str, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "parameters": dict(parameters),
        **deepcopy(PENDING_ENTRY_TEMPLATE),
    }


def generate_lhs_rows(count: int, seed: int, bounds: Mapping[str, Any]) -> List[Dict[str, Any]]:
    keys = sorted(bounds.keys())
    unit = _latin_hypercube_unit(count, len(keys), seed)
    cols = {k: _lhs_values_for_key(bounds[k], unit[:, i]) for i, k in enumerate(keys)}
    return [_build_parameters_row(cols, i) for i in range(count)]


def load_pool(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def pool_entry_for_sample(pool: Mapping[str, Any], sample_id: str) -> Optional[Dict[str, Any]]:
    for entry in pool.get("entries") or []:
        if str(entry.get("id")) == sample_id:
            return dict(entry)
    return None


def is_box_fom_sample_completed(
    repo_root: Path,
    sample_id: str,
    *,
    run_id_suffix: str = DEFAULT_FOM_RUN_ID_SUFFIX,
    pool_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """True when LHS pool marks sample COMPLETED for the expected FOM run id."""
    root = Path(repo_root)
    path = Path(pool_path or DEFAULT_POOL_PATH)
    run_id = box_fom_run_id(sample_id, run_id_suffix)
    run_root = box_fom_run_root(root, sample_id, run_id_suffix)
    out: Dict[str, Any] = {
        "sample_id": sample_id,
        "run_id": run_id,
        "ready": False,
        "run_root": str(run_root).replace("\\", "/"),
        "lhs_pool": str(path).replace("\\", "/"),
    }
    pool = load_pool(path)
    entry = pool_entry_for_sample(pool, sample_id)
    if entry is None:
        out["reason"] = "missing_lhs_entry"
        return out

    status = str(entry.get("status") or "PENDING").upper()
    last_run = str(entry.get("last_run_id") or "")
    agg_status = str(entry.get("last_aggregation_status") or "")
    out["lhs_status"] = status
    out["last_run_id"] = last_run
    out["last_aggregation_status"] = agg_status

    if status == "COMPLETED" and (not last_run or last_run == run_id):
        out["ready"] = True
        out["reason"] = "lhs_pool_completed"
        return out

    agg_json = run_root / "aggregation" / "aggregation_result.json"
    if agg_json.is_file():
        try:
            agg = json.loads(agg_json.read_text(encoding="utf-8"))
            if str(agg.get("aggregation_status") or "") == "AGGREGATION_PASS":
                out["ready"] = True
                out["reason"] = "aggregation_pass_on_disk"
                return out
        except (OSError, json.JSONDecodeError):
            pass

    out["reason"] = "incomplete"
    return out


def build_pool_document(
    *,
    count: int,
    seed: int,
    existing: Optional[Mapping[str, Any]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    existing = dict(existing or {})
    entries_in = list(existing.get("entries") or [])
    by_id = {str(e.get("id")): dict(e) for e in entries_in if e.get("id")}

    bounds = _bounds_from_pool(existing) if existing else deepcopy(DEFAULT_BOX_BOUNDS)
    if entries_in and not existing.get("lhs_bounds"):
        bounds = _infer_bounds_from_entries(entries_in)

    lhs_rows = generate_lhs_rows(count, seed, bounds)
    out_entries: List[Dict[str, Any]] = []

    for idx in range(count):
        sid = box_sample_id(idx)
        if not force and sid in by_id:
            out_entries.append(by_id[sid])
            continue
        out_entries.append(_blank_entry(sid, lhs_rows[idx]))

    doc: Dict[str, Any] = {
        "shape_name": "box",
        "shape_type": "box",
        "sampling": "lhs",
        "wood_assignment": str(existing.get("wood_assignment") or "unrestricted_5x5"),
        "seed": int(seed),
        "total_samples": int(max(int(existing.get("total_samples") or 0), count)),
        "mpi_world_size": int(existing.get("mpi_world_size") or 0),
        "lhs_bounds": bounds,
        "entries": out_entries,
    }
    return doc


def write_pool(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(doc), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pool-path", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--force", action="store_true", help="Replace existing BOX entries with fresh LHS rows")
    parser.add_argument("--dry-run", action="store_true", help="Compute pool but do not write JSON")
    parser.add_argument(
        "--check-fom-ready",
        metavar="SAMPLE_ID",
        default="",
        help="Check whether a BOX FOM/M4 sample is already COMPLETED",
    )
    parser.add_argument(
        "--run-id-suffix",
        default=DEFAULT_FOM_RUN_ID_SUFFIX,
        help="FOM run id suffix used with --check-fom-ready",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check_fom_ready:
        status = is_box_fom_sample_completed(
            REPO_ROOT,
            args.check_fom_ready,
            run_id_suffix=str(args.run_id_suffix),
            pool_path=args.pool_path,
        )
        print(json.dumps(status, indent=2))
        return 0 if status.get("ready") else 1

    count = max(1, int(args.count))
    existing = load_pool(args.pool_path)
    doc = build_pool_document(count=count, seed=int(args.seed), existing=existing, force=bool(args.force))

    preserved = 0
    created = 0
    if existing.get("entries"):
        old_ids = {str(e.get("id")) for e in existing.get("entries") or []}
        for entry in doc["entries"]:
            sid = str(entry.get("id"))
            if sid in old_ids and not args.force:
                preserved += 1
            else:
                created += 1
    else:
        created = len(doc["entries"])

    if args.dry_run:
        print(
            f"BOX_LHS_DRY_RUN path={args.pool_path.as_posix()} "
            f"count={len(doc['entries'])} preserved={preserved} created={created}"
        )
        return 0

    write_pool(args.pool_path, doc)
    print(f"BOX_LHS_READY path={args.pool_path.as_posix()} count={len(doc['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
