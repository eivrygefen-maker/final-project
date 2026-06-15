#!/usr/bin/env python3
"""Generate or extend the BOX LHS pool (``ROM/box/lhs_pool.json``)."""
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

WOOD_IDS: Tuple[str, ...] = ("spruce", "cedar", "mahogany", "rosewood", "maple")

# Conservative BOX geometry bounds (pipeline-compatible keys from existing BOX LHS).
DEFAULT_BOX_BOUNDS: Dict[str, Any] = {
    "geometry.length": {"min": 0.40, "max": 0.52},
    "geometry.width": {"min": 0.32, "max": 0.42},
    "geometry.depth": {"min": 0.06, "max": 0.16},
    "geometry.top_thickness": {"min": 0.0025, "max": 0.0035},
    "geometry.hole_radius": {"min": 0.035, "max": 0.048},
    "geometry.back_thickness": {"min": 0.0028, "max": 0.0040},
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
    """Widen defaults using numeric ranges observed in existing entries."""
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
        spec = bounds[key]
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
    params["geometry.shape_type"] = "box"
    return params


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


def box_sample_cache_dir(repo_root: Path, sample_id: str) -> Path:
    return Path(repo_root) / "audio" / "app_stk_note_cache" / "box" / sample_id


def box_sample_report_json(repo_root: Path, sample_id: str) -> Path:
    return (
        Path(repo_root)
        / "audio"
        / "debug_reports"
        / "box"
        / f"app_stk_note_library_box_{sample_id}_report.json"
    )


def is_box_sample_ready(repo_root: Path, sample_id: str) -> Dict[str, Any]:
    """Return readiness info for one BOX sample cache (no STK invocation)."""
    root = Path(repo_root)
    cache_dir = box_sample_cache_dir(root, sample_id)
    report_path = box_sample_report_json(root, sample_id)
    out: Dict[str, Any] = {
        "sample_id": sample_id,
        "ready": False,
        "cache_dir": str(cache_dir).replace("\\", "/"),
        "report_json": str(report_path).replace("\\", "/"),
    }
    if not cache_dir.is_dir():
        out["reason"] = "cache_missing"
        return out

    sys.path.insert(0, str(root / "gui"))
    from stk_app_audio_service import cache_is_ready_for_fretboard  # noqa: WPS433

    if cache_is_ready_for_fretboard(cache_dir):
        out["ready"] = True
        out["reason"] = "cache_complete"
        return out

    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("readiness") == "ready_for_app_playback" and report.get("status") == "ready":
                out["ready"] = True
                out["reason"] = "report_ready"
                return out
        except (OSError, json.JSONDecodeError):
            pass

    pos_wav = cache_dir / "S6_f2.wav"
    out["reason"] = "incomplete"
    out["has_position_alias"] = pos_wav.is_file()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pool-path", type=Path, default=DEFAULT_POOL_PATH)
    parser.add_argument("--force", action="store_true", help="Replace existing BOX entries with fresh LHS rows")
    parser.add_argument("--dry-run", action="store_true", help="Compute pool but do not write JSON")
    parser.add_argument("--check-ready", metavar="SAMPLE_ID", default="", help="Check one sample cache readiness")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.check_ready:
        status = is_box_sample_ready(REPO_ROOT, args.check_ready)
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

    if not args.dry_run:
        write_pool(args.pool_path, doc)

    print(f"BOX_LHS_READY path={args.pool_path.as_posix()} count={len(doc['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
