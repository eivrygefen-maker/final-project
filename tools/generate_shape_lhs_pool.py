#!/usr/bin/env python3
"""Generate or extend shape LHS pools for M4 FOM (classic / box / acoustic)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
FEM_SCRIPTS = REPO_ROOT / "FEM" / "scripts"
M4_SCRIPTS = (
    REPO_ROOT
    / "FEM/experiments/active_domain_validation/physics_integrity/scripts"
)
if str(FEM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FEM_SCRIPTS))
if str(M4_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(M4_SCRIPTS))

from m4_shape_registry import (  # noqa: E402
    ensure_parameters_shape_type,
    infer_shape_from_lhs_path,
    lhs_bounds_for_shape,
    normalize_shape_key,
    resolve_shape_config,
    registered_shape_keys,
)
from wood_library import finalize_lhs_thickness_params  # noqa: E402
from v2_b3_m4_lhs_pool_bridge import write_lhs_pool  # noqa: E402

M4_GUITARS_ROOT = (
    REPO_ROOT
    / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
)

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


def _build_parameters_row(
    cols: Mapping[str, Sequence[Any]],
    index: int,
    *,
    shape_key: str,
) -> Dict[str, Any]:
    params = {k: cols[k][index] for k in sorted(cols.keys())}
    params = ensure_parameters_shape_type(params, shape_key=shape_key)
    return finalize_lhs_thickness_params(params)


def _blank_entry(sample_id: str, parameters: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "parameters": dict(parameters),
        **deepcopy(PENDING_ENTRY_TEMPLATE),
    }


def generate_lhs_rows(count: int, seed: int, bounds: Mapping[str, Any], *, shape_key: str) -> List[Dict[str, Any]]:
    keys = sorted(bounds.keys())
    unit = _latin_hypercube_unit(count, len(keys), seed)
    cols = {k: _lhs_values_for_key(bounds[k], unit[:, i]) for i, k in enumerate(keys)}
    return [_build_parameters_row(cols, i, shape_key=shape_key) for i in range(count)]


def load_pool(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_target_pool_size(
    *,
    shape_key: str,
    count: int,
    existing: Optional[Mapping[str, Any]] = None,
    force: bool = False,
) -> int:
    """Pool size for generation: never shrink an existing pool unless --force regen."""
    cfg = resolve_shape_config(shape_key)
    existing = dict(existing or {})
    entries_in = list(existing.get("entries") or [])
    existing_n = len(entries_in)
    total_samples = int(existing.get("total_samples") or 0)
    requested = max(int(count), 0)

    if force and existing_n == 0:
        return max(requested, cfg.default_lhs_count)
    if force:
        return max(requested, cfg.default_lhs_count)

    if existing_n > 0 or total_samples > 0:
        return max(existing_n, total_samples, requested, cfg.default_lhs_count)
    return max(requested, cfg.default_lhs_count)


def build_pool_document(
    *,
    shape_key: str,
    count: int,
    seed: int,
    existing: Optional[Mapping[str, Any]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    cfg = resolve_shape_config(shape_key)
    existing = dict(existing or {})
    entries_in = list(existing.get("entries") or [])
    by_id = {str(e.get("id")): dict(e) for e in entries_in if e.get("id")}

    bounds = dict(existing.get("lhs_bounds") or lhs_bounds_for_shape(shape_key))
    if not bounds:
        raise ValueError(f"no LHS bounds for shape {shape_key!r}; use regenerate_lhs_pool for classic")

    target_count = resolve_target_pool_size(
        shape_key=shape_key,
        count=count,
        existing=existing,
        force=force,
    )
    lhs_rows = generate_lhs_rows(target_count, seed, bounds, shape_key=shape_key)
    out_entries: List[Dict[str, Any]] = []

    for idx in range(target_count):
        sid = cfg.sample_id(idx)
        if not force and sid in by_id:
            out_entries.append(by_id[sid])
            continue
        out_entries.append(_blank_entry(sid, lhs_rows[idx]))

    known_ids = {cfg.sample_id(i) for i in range(target_count)}
    for sid in sorted(by_id.keys()):
        if sid not in known_ids:
            out_entries.append(by_id[sid])

    return {
        "shape_name": cfg.shape_key,
        "shape_type": cfg.shape_key,
        "sampling": "lhs",
        "wood_assignment": str(existing.get("wood_assignment") or "unrestricted_5x5"),
        "seed": int(seed),
        "total_samples": int(max(int(existing.get("total_samples") or 0), target_count)),
        "mpi_world_size": int(existing.get("mpi_world_size") or 0),
        "lhs_bounds": bounds,
        "entries": out_entries,
    }


def write_pool(path: Path, doc: Mapping[str, Any], *, explicit_regeneration: bool = False) -> None:
    write_lhs_pool(path, doc, explicit_lhs_regeneration=explicit_regeneration)


def fom_run_id(sample_id: str, run_id_suffix: str) -> str:
    return f"{sample_id}_{run_id_suffix}"


def is_fom_sample_completed(
    repo_root: Path,
    shape_key: str,
    sample_id: str,
    *,
    run_id_suffix: str,
    pool_path: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg = resolve_shape_config(shape_key)
    root = Path(repo_root)
    path = Path(pool_path or cfg.lhs_pool_path(root))
    run_id = fom_run_id(sample_id, run_id_suffix)
    run_root = M4_GUITARS_ROOT / sample_id / "runs" / run_id
    out: Dict[str, Any] = {
        "shape_name": cfg.shape_key,
        "sample_id": sample_id,
        "run_id": run_id,
        "ready": False,
        "run_root": str(run_root).replace("\\", "/"),
        "lhs_pool": str(path).replace("\\", "/"),
    }
    pool = load_pool(path)
    entry = None
    for row in pool.get("entries") or []:
        if str(row.get("id")) == sample_id:
            entry = dict(row)
            break
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


def generate_classic_via_regenerate(count: int, seed: int, *, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(FEM_SCRIPTS / "regenerate_lhs_pool.py"),
        "--shape",
        "classic",
        "--samples",
        str(count),
        "--seed",
        str(seed),
    ]
    if dry_run:
        print(f"CLASSIC_LHS_DRY_RUN would_run={' '.join(cmd)}")
        return 0
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", required=True, choices=registered_shape_keys())
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--pool-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-classic-regen",
        action="store_true",
        help="Permit regenerating ROM/classic/lhs_pool.json (default: refused).",
    )
    parser.add_argument(
        "--ensure-existing",
        action="store_true",
        help="Extend missing pool rows only; never shrink entries (safe before batch runs).",
    )
    parser.add_argument("--check-fom-ready", metavar="SAMPLE_ID", default="")
    parser.add_argument("--run-id-suffix", default="m4prod1")
    args = parser.parse_args(list(argv) if argv is not None else None)

    shape_key = normalize_shape_key(args.shape)
    cfg = resolve_shape_config(shape_key)
    pool_path = Path(args.pool_path or cfg.lhs_pool_path(REPO_ROOT))

    if args.check_fom_ready:
        status = is_fom_sample_completed(
            REPO_ROOT,
            shape_key,
            args.check_fom_ready,
            run_id_suffix=str(args.run_id_suffix),
            pool_path=pool_path,
        )
        print(json.dumps(status, indent=2))
        return 0 if status.get("ready") else 1

    count = int(args.count if args.count is not None else cfg.default_lhs_count)

    if shape_key == "classic" and not args.allow_classic_regen:
        if pool_path.is_file():
            print(
                f"CLASSIC_LHS_SKIP path={pool_path.as_posix()} "
                "(existing pool preserved; pass --allow-classic-regen to replace)"
            )
            return 0
        print("error: classic pool missing; use FEM/scripts/regenerate_lhs_pool.py or --allow-classic-regen", file=sys.stderr)
        return 2

    if shape_key == "classic":
        return generate_classic_via_regenerate(count, int(args.seed), dry_run=bool(args.dry_run))

    existing = load_pool(pool_path)
    if args.ensure_existing and existing:
        count = resolve_target_pool_size(
            shape_key=shape_key,
            count=0,
            existing=existing,
            force=False,
        )
    doc = build_pool_document(
        shape_key=shape_key,
        count=count,
        seed=int(args.seed),
        existing=existing,
        force=bool(args.force),
    )

    if args.dry_run:
        print(
            f"SHAPE_LHS_DRY_RUN shape={shape_key} path={pool_path.as_posix()} "
            f"count={len(doc['entries'])}"
        )
        return 0

    write_pool(pool_path, doc, explicit_regeneration=bool(args.force))
    print(f"SHAPE_LHS_READY shape={shape_key} path={pool_path.as_posix()} count={len(doc['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
