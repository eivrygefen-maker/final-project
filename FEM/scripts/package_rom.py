#!/usr/bin/env python3
"""
Package MMR-selected modes from ``selected_modes.csv`` into a single NPZ ROM file
and optionally reset SORTING scratch data for the next run.

Mode columns are read as CSR sparse (``*.smx.npz`` from workers) or legacy dense ``.npy``,
aggregated with ``scipy.sparse.hstack``, and written as one bundled compressed NPZ
containing CSR arrays (``ev_data``, ``ev_indices``, ``ev_indptr``, ``ev_shape``)
plus ``frequencies``, ``wood_participations``, and pipeline provenance metadata.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fem_harvest_filter import HARVEST_FILTER_POLICY_VERSION
from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX, load_mode_column_any

FEM_ROOT = SCRIPT_DIR.parent
SORTING_ROOT = FEM_ROOT / "SORTING"


def _default_csv() -> Path:
    return SORTING_ROOT / "selected_modes.csv"


def _default_npz() -> Path:
    return SORTING_ROOT / "final_guitar_rom.npz"


def _load_vector_path_map(candidates_path: Path) -> Dict[int, str]:
    if not candidates_path.is_file():
        return {}
    try:
        data = json.loads(candidates_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: Dict[int, str] = {}
    for c in data.get("candidates", []) or []:
        try:
            mid = int(c.get("id"))
            vp = c.get("vector_path")
            if isinstance(vp, str) and vp.strip():
                out[mid] = vp.strip().replace("\\", "/")
        except (TypeError, ValueError):
            continue
    return out


def _load_pipeline_meta(candidates_path: Path) -> Dict[str, object]:
    if not candidates_path.is_file():
        return {}
    try:
        data = json.loads(candidates_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    meta = data.get("pipeline_meta")
    return dict(meta) if isinstance(meta, dict) else {}


def _resolve_mode_vector_path(
    mode_id: int,
    temp_modes: Path,
    sorting_root: Path,
    path_by_id: Dict[int, str],
) -> Path:
    primary = temp_modes / f"mode_{mode_id:06d}{MODE_VECTOR_FILE_SUFFIX}"
    if primary.is_file():
        return primary
    legacy = temp_modes / f"mode_{mode_id:06d}.npy"
    if legacy.is_file():
        return legacy
    rel = path_by_id.get(mode_id)
    if rel:
        cand = (sorting_root / rel).resolve()
        if cand.is_file():
            return cand
        cand2 = (temp_modes / Path(rel).name).resolve()
        if cand2.is_file():
            return cand2
    return primary


def _resolve_sorting_root(csv_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    parent = csv_path.parent.resolve()
    if (parent / "temp_modes").is_dir():
        return parent
    return SORTING_ROOT.resolve()


def _read_winners(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Selected modes CSV not found: {csv_path}")
    rows: List[Dict[str, object]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        fields = {h.strip().lower(): h for h in reader.fieldnames}
        for key in ("id", "hz", "wood_participation"):
            if key not in fields:
                raise ValueError(f"CSV missing required column '{key}'. Found: {list(reader.fieldnames)}")
        id_h, hz_h, wood_h = fields["id"], fields["hz"], fields["wood_participation"]
        st_h = fields.get("source_target_hz")
        pol_h = fields.get("harvest_filter_policy")
        cls_h = fields.get("harvest_class")
        for rec in reader:
            try:
                row: Dict[str, object] = {
                    "id": int(float(rec[id_h])),
                    "hz": float(rec[hz_h]),
                    "wood": float(rec[wood_h]),
                }
                if st_h and rec.get(st_h, "").strip():
                    row["source_target_hz"] = float(rec[st_h])
                if pol_h and rec.get(pol_h, "").strip():
                    row["harvest_filter_policy"] = str(rec[pol_h]).strip()
                if cls_h and rec.get(cls_h, "").strip():
                    row["harvest_class"] = str(rec[cls_h]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Bad CSV row {rec!r}: {exc}") from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")
    return rows


def _cleanup_workspace(sorting_root: Path, keep_csv: Path, fem_outputs_modes: Path) -> None:
    temp_modes = sorting_root / "temp_modes"
    temp_results = sorting_root / "temp_results"
    candidates_log = sorting_root / "candidates_log.json"

    removed_vec = 0
    if temp_modes.is_dir():
        for pattern in ("mode_*.npy", f"mode_*{MODE_VECTOR_FILE_SUFFIX}"):
            for p in temp_modes.glob(pattern):
                try:
                    os.remove(str(p))
                    removed_vec += 1
                except OSError:
                    pass

    removed_json = 0
    if temp_results.is_dir():
        for p in temp_results.rglob("*.json"):
            try:
                os.remove(str(p))
                removed_json += 1
            except OSError:
                pass

    try:
        os.remove(str(candidates_log))
    except OSError:
        pass

    removed_export = 0
    if fem_outputs_modes.is_dir():
        for pattern in ("*.vtk", "*.h5", "*.xdmf"):
            for p in fem_outputs_modes.glob(pattern):
                try:
                    os.remove(str(p))
                    removed_export += 1
                except OSError:
                    pass
        raw_npz = fem_outputs_modes / "coupled_modes_raw.npz"
        try:
            if raw_npz.is_file():
                os.remove(str(raw_npz))
                removed_export += 1
        except OSError:
            pass

    print(
        f"Cleanup: removed {removed_vec} mode vector file(s) under temp_modes/, "
        f"{removed_json} file(s) under temp_results/, "
        f"and candidates_log.json (if present); "
        f"{removed_export} FEM/outputs/modes_3d artifact(s). Kept: {keep_csv.name}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Package selected_modes.csv into final_guitar_rom.npz")
    parser.add_argument("--csv", type=Path, default=_default_csv(), help="Path to selected_modes.csv")
    parser.add_argument("--out", type=Path, default=_default_npz(), help="Output NPZ path")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete temp_modes vectors, temp_results/*.json, and candidates_log.json (keeps selected_modes.csv).",
    )
    parser.add_argument(
        "--sorting-root",
        type=Path,
        default=None,
        help=(
            "Folder containing temp_modes/ and candidates_log.json (default: parent of --csv if it has "
            "temp_modes/, else FEM/SORTING next to this script)."
        ),
    )
    args = parser.parse_args()

    csv_path = args.csv.resolve()
    out_path = args.out.resolve()
    sorting_root = _resolve_sorting_root(csv_path, args.sorting_root)
    temp_modes = sorting_root / "temp_modes"
    candidates_path = sorting_root / "candidates_log.json"
    path_by_id = _load_vector_path_map(candidates_path)
    pipeline_meta = _load_pipeline_meta(candidates_path)

    try:
        winners = _read_winners(csv_path)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    cols: List[sparse.csr_matrix] = []
    freqs: List[float] = []
    woods: List[float] = []
    source_targets: List[float] = []
    harvest_policies: List[str] = []
    harvest_classes: List[str] = []
    missing: List[int] = []

    for row in winners:
        mid = int(row["id"])
        hz = float(row["hz"])
        wood = float(row["wood"])
        vec_path = _resolve_mode_vector_path(mid, temp_modes, sorting_root, path_by_id)
        if not vec_path.is_file():
            missing.append(mid)
            continue
        try:
            col = load_mode_column_any(vec_path)
        except Exception as exc:
            print(f"Error loading {vec_path}: {exc}", file=sys.stderr)
            return 1
        cols.append(col)
        freqs.append(hz)
        woods.append(wood)
        st = row.get("source_target_hz")
        source_targets.append(float(st) if st is not None else float("nan"))
        harvest_policies.append(
            str(row.get("harvest_filter_policy") or pipeline_meta.get("harvest_filter_policy") or HARVEST_FILTER_POLICY_VERSION)
        )
        harvest_classes.append(str(row.get("harvest_class") or ""))

    if missing:
        print(
            f"Error: missing mode vector for id(s): {missing[:20]}{'...' if len(missing) > 20 else ''} "
            f"(expected *{MODE_VECTOR_FILE_SUFFIX} or .npy under temp_modes/)",
            file=sys.stderr,
        )
        return 1

    lengths = {c.shape[0] for c in cols}
    if len(lengths) != 1:
        print(f"Error: inconsistent eigenvector lengths: {sorted(lengths)}", file=sys.stderr)
        return 1

    eigenvectors = sparse.hstack(cols, format="csr").astype(np.float32, copy=False)
    del cols
    gc.collect()

    frequencies = np.asarray(freqs, dtype=np.float64)
    wood_participations = np.asarray(woods, dtype=np.float64)
    source_target_hz = np.asarray(source_targets, dtype=np.float64)
    policy_arr = np.asarray(harvest_policies, dtype="U64")
    class_arr = np.asarray(harvest_classes, dtype="U32")

    sweep_lo = float(pipeline_meta.get("sweep_hz_min", 60.0))
    sweep_hi = float(pipeline_meta.get("sweep_hz_max", 550.0))
    policy_version = str(
        pipeline_meta.get("harvest_filter_policy", HARVEST_FILTER_POLICY_VERSION)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        ev_data=eigenvectors.data,
        ev_indices=eigenvectors.indices,
        ev_indptr=eigenvectors.indptr,
        ev_shape=np.asarray(eigenvectors.shape, dtype=np.int64),
        frequencies=frequencies,
        wood_participations=wood_participations,
        source_target_hz=source_target_hz,
        harvest_filter_policy=policy_arr,
        harvest_class=class_arr,
        pipeline_harvest_filter_policy=np.asarray(policy_version, dtype="U64"),
        pipeline_sweep_hz_min=np.asarray(sweep_lo, dtype=np.float64),
        pipeline_sweep_hz_max=np.asarray(sweep_hi, dtype=np.float64),
        pipeline_coupled_worker_num_modes_cap=np.asarray(
            int(pipeline_meta.get("coupled_worker_num_modes_cap", 40)), dtype=np.int32
        ),
    )

    try:
        with np.load(str(out_path), allow_pickle=False) as z:
            need = (
                "ev_data",
                "ev_indices",
                "ev_indptr",
                "ev_shape",
                "frequencies",
                "wood_participations",
                "source_target_hz",
                "pipeline_harvest_filter_policy",
            )
            if any(k not in z.files for k in need):
                print(f"Error: output NPZ missing CSR/metadata keys: {out_path}", file=sys.stderr)
                return 1
            shape = tuple(int(x) for x in np.asarray(z["ev_shape"]).ravel())
            ev_chk = sparse.csr_matrix(
                (z["ev_data"], z["ev_indices"], z["ev_indptr"]),
                shape=shape,
                dtype=np.float32,
            )
            if ev_chk.dtype != np.dtype(np.float32):
                print(f"Error: expected float32 CSR data, got {ev_chk.dtype}", file=sys.stderr)
                return 1
            if ev_chk.shape != eigenvectors.shape:
                print(
                    f"Error: eigenvectors shape mismatch after save {ev_chk.shape} vs {eigenvectors.shape}",
                    file=sys.stderr,
                )
                return 1
            if int(ev_chk.nnz) != int(eigenvectors.nnz):
                print(f"Error: nnz mismatch after save {ev_chk.nnz} vs {eigenvectors.nnz}", file=sys.stderr)
                return 1
    except OSError as exc:
        print(f"Error: could not verify saved NPZ {out_path}: {exc}", file=sys.stderr)
        return 1

    gc.collect()

    nbytes = out_path.stat().st_size if out_path.is_file() else 0
    nrows, ncols = eigenvectors.shape
    nnz = int(eigenvectors.nnz)
    sparsity = 1.0 - (nnz / max(float(nrows * ncols), 1.0))
    print(
        f"Created ROM archive: {out_path}\n"
        f"  eigenvectors CSR: shape {eigenvectors.shape}, nnz={nnz}, sparsity≈{sparsity:.4f}\n"
        f"  frequencies: {frequencies.shape}  |  wood_participations: {wood_participations.shape}\n"
        f"  provenance: policy={policy_version!r} sweep=[{sweep_lo:.0f},{sweep_hi:.0f}] Hz\n"
        f"  file size: {nbytes / (1024 * 1024):.2f} MiB ({nbytes} bytes)"
    )

    if args.cleanup:
        _cleanup_workspace(sorting_root, csv_path, FEM_ROOT / "outputs" / "modes_3d")
        gc.collect()
        print("Workspace reset complete; selected_modes.csv was preserved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
