#!/usr/bin/env python3
"""
Inspect coupled mode vectors: structural (u) vs acoustic (p) L2 energy split.

Worker ``*.smx.npz`` files store the **full** SLEPc mixed eigenvector column
(block ordering [u; p] with n_u displacement DOFs then n_p pressure DOFs).

Usage:
  python FEM/scripts/inspect_mode_vectors.py FEM/SORTING/temp_modes/mode_w_102000_000.smx.npz
  python FEM/scripts/inspect_mode_vectors.py path/to/mode.smx.npz --n-u 228284 --n-p 228284
  python FEM/scripts/inspect_mode_vectors.py --glob "FEM/SORTING/temp_modes/*.smx.npz" --n-u 228284
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import sparse

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fem_mode_array_utils import csr_col_norm, load_mode_column_any  # noqa: E402

# Mirror fem_main_3d.py facet / BC protocol (for audit printout).
TAG_TOP = 1
TAG_SOUNDHOLE = 2
TAG_BACK = 3
RIBS_SURFACE_TAG = 4
WOOD_FIX_SURFACE_TAG = 5


def _dense_column(path: Path) -> np.ndarray:
    m = load_mode_column_any(path)
    return np.asarray(m.toarray(), dtype=np.float64).reshape(-1)


def _l2_norm(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(v))


def _split_up(
    x: np.ndarray, n_u: int, n_p: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    n = int(x.size)
    nu = int(n_u)
    if nu <= 0 or nu > n:
        raise ValueError(f"Invalid n_u={nu} for vector length {n}")
    if n_p is not None:
        np_ = int(n_p)
        if nu + np_ != n:
            raise ValueError(f"n_u + n_p = {nu + np_} != vector length {n}")
    else:
        np_ = n - nu
    u_vec = x[:nu]
    p_vec = x[nu : nu + np_]
    return u_vec, p_vec, nu, np_


def _load_meta_npz(meta_path: Path) -> Tuple[Optional[int], Optional[int]]:
    data = np.load(str(meta_path), allow_pickle=False)
    keys = set(data.files)
    n_u = int(data["n_u"]) if "n_u" in keys else None
    n_p = int(data["n_p"]) if "n_p" in keys else None
    return n_u, n_p


def _guess_n_u_n_p(n: int) -> Tuple[Optional[int], Optional[int]]:
    if n % 2 == 0:
        half = n // 2
        return half, half
    return None, None


def inspect_one(path: Path, n_u: Optional[int], n_p: Optional[int]) -> int:
    x = _dense_column(path)
    n = x.size

    if n_u is None:
        n_u_g, n_p_g = _guess_n_u_n_p(n)
        if n_u_g is None:
            print(f"[{path.name}] length={n} — provide --n-u (and optionally --n-p).", file=sys.stderr)
            return 1
        n_u, n_p = n_u_g, n_p_g
        meta_src = "assumed 50/50 split"
    else:
        meta_src = "CLI/meta"
        if n_p is None:
            n_p = n - int(n_u)

    try:
        u_vec, p_vec, nu, np_ = _split_up(x, int(n_u), int(n_p) if n_p is not None else None)
    except ValueError as exc:
        print(f"[{path.name}] split failed: {exc}", file=sys.stderr)
        return 1

    norm_u = _l2_norm(u_vec)
    norm_p = _l2_norm(p_vec)
    norm_x = _l2_norm(x)
    ratio_up = norm_u / max(norm_p, 1.0e-30)
    ratio_pu = norm_p / max(norm_u, 1.0e-30)
    frac_u = (norm_u * norm_u) / max(norm_x * norm_x, 1.0e-30)

    print(f"\n=== {path} ===")
    print(f"  meta: {meta_src}  n_u={nu}, n_p={np_}, n_total={n}")
    print(f"  ||x||_2     = {norm_x:.6e}")
    print(f"  ||u_vec||_2 = {norm_u:.6e}  (structural block, rows 0..{nu - 1})")
    print(f"  ||p_vec||_2 = {norm_p:.6e}  (acoustic block, rows {nu}..{nu + np_ - 1})")
    print(f"  ||u||/||p||  = {ratio_up:.6e}")
    print(f"  ||p||/||u||  = {ratio_pu:.6e}")
    print(f"  fraction of ||x||^2 in u  = {frac_u:.4f}")

    if norm_u < 1.0e-12 and norm_p > 1.0e-6:
        print("  [VERDICT] PURE ACOUSTIC — structural block ~0 (matches 0 wood participation).")
    elif norm_p < 1.0e-12 and norm_u > 1.0e-6:
        print("  [VERDICT] PURE STRUCTURAL — pressure block ~0.")
    elif norm_u > 1.0e-6 and norm_p > 1.0e-6:
        print("  [VERDICT] COUPLED — both blocks active.")
    else:
        print("  [VERDICT] WEAK / NUMERIC — both blocks very small.")

    if n == nu:
        print(
            "  [note] Vector length equals n_u only — file may store u-displacement "
            "slice, not full [u;p] (worker normally saves full mixed column)."
        )

    # Worker gate uses u-block norm only
    m = load_mode_column_any(path)
    print(f"  worker u-slice ||u||_2 (first {nu} rows) = {csr_col_norm(m[:nu, :]):.6e}")
    return 0


def print_bc_audit() -> None:
    print("\n=== fem_main_3d.py Dirichlet BC audit (coupled path) ===")
    print("  Pressure gauge: P=0 on facet tag 2 (soundhole) only.")
    print(f"  Displacement clamp: u=0 on facet tag {RIBS_SURFACE_TAG} (Ribs_Sides) ONLY.")
    print(f"  Tags {TAG_TOP} (top), {TAG_BACK} (back): NO displacement DirichletBC.")
    print(f"  Tag {WOOD_FIX_SURFACE_TAG} (wood_fix): NOT used for BCs in coupled solve.")
    print("  Structural-only diagnostic branch: free–free (no u constraints).")
    print("  If ||u|| is ~0 in full eigenvectors, cause is likely EPS mode shape (pressure-dominated),")
    print("  not accidental clamp on tags 1/3.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect u vs p norms in mode vector files.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Mode file(s): *.smx.npz or legacy .npy",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="",
        help='Glob under repo root, e.g. "FEM/SORTING/temp_modes/*.smx.npz"',
    )
    parser.add_argument("--n-u", type=int, default=None, help="Structural DOF count (global)")
    parser.add_argument("--n-p", type=int, default=None, help="Pressure DOF count (global)")
    parser.add_argument(
        "--meta-npz",
        type=Path,
        default=None,
        help="coupled_modes_raw.npz with n_u, n_p fields",
    )
    parser.add_argument("--bc-audit", action="store_true", help="Print BC tag summary from fem_main_3d")
    args = parser.parse_args()

    if args.bc_audit:
        print_bc_audit()

    n_u, n_p = args.n_u, args.n_p
    if args.meta_npz is not None:
        mp = args.meta_npz.resolve()
        if not mp.is_file():
            print(f"meta npz not found: {mp}", file=sys.stderr)
            return 1
        mu, mp_ = _load_meta_npz(mp)
        n_u = mu if n_u is None else n_u
        n_p = mp_ if n_p is None else n_p
        print(f"[meta] loaded from {mp}: n_u={n_u}, n_p={n_p}")

    files: list[Path] = [p.resolve() for p in args.paths]
    if args.glob:
        files.extend(sorted(REPO_ROOT.glob(args.glob)))
    if not files:
        default = REPO_ROOT / "FEM" / "SORTING" / "temp_modes"
        if default.is_dir():
            files = sorted(default.glob("*.smx.npz"))[:5]
        if not files:
            parser.error("No mode files given. Pass a path or --glob.")

    rc = 0
    for fp in files:
        if not fp.is_file():
            print(f"missing: {fp}", file=sys.stderr)
            rc = 1
            continue
        rc = max(rc, inspect_one(fp, n_u, n_p))

    # Optional: read n_u from a worker result JSON if present
    if n_u is None and len(files) == 1:
        try:
            res_dir = files[0].parents[1] / "temp_results"
            if res_dir.is_dir():
                for rj in sorted(res_dir.glob("result_*.json"))[:1]:
                    with open(rj, encoding="utf-8") as f:
                        blob = json.load(f)
                    print(f"\n[hint] nearest result json: {rj} (modes={blob.get('num_modes_returned')})")
        except Exception:
            pass

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
