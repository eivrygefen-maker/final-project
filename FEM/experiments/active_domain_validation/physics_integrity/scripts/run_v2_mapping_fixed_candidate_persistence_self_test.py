#!/usr/bin/env python3
"""
No-EVP persistence self-test for mapping-fixed candidate vector save/load (mpiexec -n 1).

Must pass before any replacement mapping-corrected baseline eigensolve is authorized.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from v2_mesh_convergence_common import CONV_DIAG, load_manifest, mesh_path, sample_spec_from_case, solve_case_dir, write_json
from v2_mapping_fixed_candidate_persistence import candidate_slot_path, persist_candidate_bank

OUT_JSON = CONV_DIAG / "v2_mapping_fixed_candidate_persistence_self_test.json"
OUT_MD = CONV_DIAG / "v2_mapping_fixed_candidate_persistence_self_test.md"
CASE_ID = "baseline_coupled_v2"
SCRATCH_SUBDIR = "_persistence_self_test_scratch"


def _pressure_mac(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(abs(np.vdot(a, b)) / (na * nb))


def main() -> int:
    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[persistence_self_test] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    mesh_file = mesh_path("L_mid", CASE_ID)
    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_meta_path = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    seed_meta = (
        json.loads(seed_meta_path.read_text(encoding="utf-8")) if seed_meta_path.is_file() else {}
    )
    scratch = CONV_DIAG / SCRATCH_SUBDIR
    if scratch.is_dir():
        for p in scratch.rglob("*.smx.npz"):
            try:
                p.unlink()
            except OSError:
                pass
    scratch.mkdir(parents=True, exist_ok=True)

    checks: List[Dict[str, Any]] = []
    passed = True

    if not seed_npy.is_file():
        checks.append({"check": "seed_file_exists", "pass": False})
        passed = False
    else:
        seed = np.asarray(np.load(str(seed_npy)), dtype=np.float64).ravel()
        seed_norm = float(np.linalg.norm(seed))
        checks.append(
            {
                "check": "seed_file_exists",
                "pass": True,
                "seed_vector_length": int(seed.size),
                "seed_vector_norm": seed_norm,
            }
        )
        bank_rec = [
            {
                "eps_slot_index": 0,
                "candidate_index": 0,
                "mu_raw": None,
                "lam_phys": None,
                "reported_frequency_hz": float(seed_meta.get("locator_frequency_hz", float("nan"))),
                "sigma_used_hz": None,
                "st_type": "self_test",
                "eps_eigenvalue_semantics": "slepc_backtransformed",
                "legacy_double_shift_mapping_disabled": True,
                "vector": seed,
            }
        ]

        def _save_only(vec: np.ndarray, mode_path: Path, rec: Dict[str, Any]) -> Dict[str, Any]:
            from fem_mode_array_utils import dense_to_csr_f32_column, save_mode_csr

            save_mode_csr(mode_path, dense_to_csr_f32_column(vec))
            return {"persistence_status": "saved"}

        n_saved, _rows, save_errors = persist_candidate_bank(
            scratch, bank_rec, save_vector_fn=_save_only
        )
        slot_path = candidate_slot_path(scratch / "modes", 0)
        length_ok = int(seed.size) == int(seed.size)
        reload_ok = False
        norm_ok = False
        mac_ok = False
        rayleigh_ok = False
        xH_Mx_ok = False
        reload_len = None
        reload_norm = float("nan")
        mac = float("nan")
        rayleigh_f = float("nan")
        xH_Mx = float("nan")
        seed_f = float(seed_meta.get("locator_frequency_hz", float("nan")))

        if n_saved != 1 or save_errors:
            checks.append(
                {
                    "check": "persist_candidate_bank",
                    "pass": False,
                    "n_saved": n_saved,
                    "save_errors": save_errors,
                    "vector_file": str(slot_path),
                }
            )
            passed = False
        elif not slot_path.is_file():
            checks.append({"check": "vector_file_written", "pass": False, "path": str(slot_path)})
            passed = False
        else:
            from fem_mode_array_utils import load_mode_column_any

            reloaded = np.asarray(load_mode_column_any(slot_path).toarray(), dtype=np.float64).ravel()
            reload_len = int(reloaded.size)
            reload_norm = float(np.linalg.norm(reloaded))
            length_ok = reload_len == int(seed.size)
            norm_ok = math.isfinite(reload_norm) and math.isfinite(seed_norm)
            if norm_ok:
                norm_ok = abs(reload_norm - seed_norm) / max(seed_norm, 1.0e-30) <= 1.0e-5
            reload_ok = length_ok and norm_ok

            p_to_W = np.asarray(seed_meta.get("p_to_W") or [], dtype=np.int32).ravel()
            if p_to_W.size == 0:
                checks.append({"check": "pressure_mac", "pass": False, "reason": "p_to_W_missing_in_meta"})
                passed = False
            else:
                mac = _pressure_mac(seed[p_to_W], reloaded[p_to_W])
                mac_ok = math.isfinite(mac) and mac >= 0.999
                checks.append(
                    {
                        "check": "pressure_mac",
                        "pass": mac_ok,
                        "pressure_MAC": mac,
                        "threshold": 0.999,
                    }
                )
                if not mac_ok:
                    passed = False

            try:
                from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay
                from physical_fsi_seed_residual_audit import _rayleigh_metrics

                sample = sample_spec_from_case(case)
                A, M, cfg = _assemble_reduced_coupled_replay(
                    mesh_file, sample, coupling_enabled=True
                )
                try:
                    for vec_label, vec in (("seed", seed), ("reloaded", reloaded)):
                        ray = _rayleigh_metrics(A, M, vec, seed_f_hz=seed_f)
                        if vec_label == "seed":
                            xH_Mx = float(ray.get("xH_Mx", float("nan")))
                            rayleigh_f = float(ray.get("rayleigh_f_hz", float("nan")))
                    xH_Mx_ok = math.isfinite(xH_Mx) and abs(xH_Mx) > 1.0e-30
                    if math.isfinite(seed_f) and seed_f > 0 and math.isfinite(rayleigh_f):
                        rayleigh_ok = abs(rayleigh_f - seed_f) / seed_f <= 0.01
                    checks.append(
                        {
                            "check": "seed_replay_rayleigh",
                            "pass": xH_Mx_ok and rayleigh_ok,
                            "xH_Mx": xH_Mx,
                            "rayleigh_f_hz": rayleigh_f,
                            "seed_f_hz": seed_f,
                        }
                    )
                    if not (xH_Mx_ok and rayleigh_ok):
                        passed = False
                finally:
                    A.destroy()
                    M.destroy()
            except Exception as exc:
                checks.append(
                    {
                        "check": "seed_replay_rayleigh",
                        "pass": False,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
                passed = False

            checks.append(
                {
                    "check": "vector_length_preserved",
                    "pass": length_ok,
                    "original_length": int(seed.size),
                    "reloaded_length": reload_len,
                }
            )
            checks.append(
                {
                    "check": "vector_norm_preserved",
                    "pass": norm_ok,
                    "original_norm": seed_norm,
                    "reloaded_norm": reload_norm,
                }
            )
            if not length_ok or not norm_ok:
                passed = False

    report: Dict[str, Any] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_type": "no_evp_mapping_fixed_candidate_persistence_self_test",
        "seed_npy": str(seed_npy),
        "scratch_dir": str(scratch),
        "candidate_vector_path_template": str(scratch / "modes" / "candidate_eps_slot_0000.smx.npz"),
        "checks": checks,
        "self_test_pass": bool(passed),
        "authorizes_replacement_baseline_eigensolve": bool(passed),
    }
    write_json(OUT_JSON, report)

    lines = [
        "# Mapping-fixed candidate persistence self-test",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        f"**self_test_pass:** `{report['self_test_pass']}`",
        "",
        f"**authorizes_replacement_baseline_eigensolve:** `{report['authorizes_replacement_baseline_eigensolve']}`",
        "",
        "## Checks",
        "",
    ]
    for c in checks:
        lines.append(f"- {c.get('check')}: pass={c.get('pass')}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[persistence_self_test] pass={passed} wrote {OUT_JSON}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
