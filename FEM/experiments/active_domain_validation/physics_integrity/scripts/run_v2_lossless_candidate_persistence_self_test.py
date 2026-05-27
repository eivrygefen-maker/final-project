#!/usr/bin/env python3
"""
No-EPS validation of diagnostic lossless vs lossy mode persistence (seed + synthetic).

Does not run EPS or modify existing replacement baseline artifacts.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fem_mode_array_utils import (
    MODE_VECTOR_RELATIVE_EPS,
    dense_to_csr_f32_column,
    load_mode_column_any,
    load_mode_dense_f64_lossless,
    save_mode_csr,
    save_mode_dense_f64_lossless,
)
from v2_mesh_convergence_common import CONV_DIAG, case_by_id, load_manifest, mesh_path, sample_spec_from_case, solve_case_dir

OUT_JSON = CONV_DIAG / "v2_lossless_candidate_persistence_self_test.json"
CASE_ID = "baseline_coupled_v2"
N_W = 277626


def _rayleigh_on_dense(A, M, x: np.ndarray) -> Dict[str, float]:
    from physical_fsi_seed_residual_audit import _rayleigh_metrics

    return _rayleigh_metrics(A, M, x, seed_f_hz=float("nan"))


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[lossless_self_test] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    manifest = load_manifest()
    case = case_by_id(manifest, CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    sample = sample_spec_from_case(case)
    mesh_file = mesh_path("L_mid", CASE_ID)

    from v2_unreg_offset_report_evaluator import assemble_replay_operators

    A, M, _u, _p, _meta = assemble_replay_operators(
        mesh_file, sample, out_dir=case_dir / "lossless_self_test_scratch"
    )
    checks: Dict[str, Any] = {}
    try:
        seed = np.asarray(np.load(str(seed_npy)), dtype=np.float64).ravel()
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            lossless_p = td_path / "seed.smx.dense.npy"
            sparse_p = td_path / "seed.smx.npz"
            save_mode_dense_f64_lossless(lossless_p, seed)
            save_mode_csr(sparse_p, dense_to_csr_f32_column(seed))
            seed_rt = load_mode_dense_f64_lossless(lossless_p)
            norm_err = float(np.linalg.norm(seed_rt - seed) / max(np.linalg.norm(seed), 1e-30))
            checks["seed_lossless_round_trip_norm_relative_error"] = norm_err
            checks["seed_lossless_round_trip_pass"] = norm_err < 1e-12
            ray_seed = _rayleigh_on_dense(A, M, seed)
            ray_rt = _rayleigh_on_dense(A, M, seed_rt)
            checks["seed_xH_Mx_original"] = float(ray_seed.get("xH_Mx", float("nan")))
            checks["seed_xH_Mx_lossless_reload"] = float(ray_rt.get("xH_Mx", float("nan")))
            checks["seed_rayleigh_pass"] = (
                math.isfinite(checks["seed_xH_Mx_original"])
                and abs(checks["seed_xH_Mx_original"]) > 1e-30
                and math.isfinite(checks["seed_xH_Mx_lossless_reload"])
            )

            # Synthetic: many sub-threshold components relative to peak
            synth = np.zeros(N_W, dtype=np.float64)
            synth[0] = 1.0
            for k in range(1, 200):
                synth[k] = MODE_VECTOR_RELATIVE_EPS * 0.1 * (1.0 + 0.01 * k)
            for k in range(24000, 24039):
                synth[k] = 0.05 * MODE_VECTOR_RELATIVE_EPS
            synth_p = td_path / "synth.smx.dense.npy"
            synth_sparse = td_path / "synth.smx.npz"
            save_mode_dense_f64_lossless(synth_p, synth)
            save_mode_csr(synth_sparse, dense_to_csr_f32_column(synth))
            synth_lossless = load_mode_dense_f64_lossless(synth_p)
            synth_loaded = np.asarray(load_mode_column_any(synth_sparse).toarray(), dtype=np.float64).ravel()
            checks["synthetic_lossless_equals_dense"] = bool(
                np.allclose(synth_lossless, synth, rtol=0, atol=0)
            )
            checks["synthetic_sparse_alters_vector"] = not np.allclose(
                synth_loaded, synth, rtol=0, atol=1e-15
            )
            checks["synthetic_sparse_nnz"] = int(
                load_mode_column_any(synth_sparse).nnz
            )
            ray_s = _rayleigh_on_dense(A, M, synth)
            ray_sl = _rayleigh_on_dense(A, M, synth_lossless)
            ray_sp = _rayleigh_on_dense(A, M, synth_loaded)
            checks["synthetic_xH_Mx_dense"] = float(ray_s.get("xH_Mx", float("nan")))
            checks["synthetic_xH_Mx_lossless"] = float(ray_sl.get("xH_Mx", float("nan")))
            checks["synthetic_xH_Mx_sparse"] = float(ray_sp.get("xH_Mx", float("nan")))
            checks["synthetic_sparse_mass_null_expected"] = (
                not math.isfinite(checks["synthetic_xH_Mx_sparse"])
                or abs(checks["synthetic_xH_Mx_sparse"]) < 1e-30
            )
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    passed = bool(
        checks.get("seed_lossless_round_trip_pass")
        and checks.get("seed_rayleigh_pass")
        and checks.get("synthetic_lossless_equals_dense")
        and checks.get("synthetic_sparse_alters_vector")
    )
    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "self_test_pass": passed,
        "checks": checks,
        "MODE_VECTOR_RELATIVE_EPS": MODE_VECTOR_RELATIVE_EPS,
        "no_eigensolve_executed": True,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[lossless_self_test] pass={passed} wrote {OUT_JSON}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
