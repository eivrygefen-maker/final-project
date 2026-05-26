#!/usr/bin/env python3
"""
Experiment-only v2_material_structural_harvest_extension.

Six coupled v2 solves on the validation mesh with uniform structural harvest 200–320 Hz,
then post-solve MAC/subspace comparison (frozen coupled_physical_core_v2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_material_structural_compare import case_spectrum_report
from v2_sensitivity_common import (
    HARVEST_EXT_DIAG,
    HARVEST_EXT_SAMPLES,
    STRUCTURAL_HARVEST_HI,
    STRUCTURAL_HARVEST_LO,
    STRUCTURAL_HARVEST_NUM_MODES,
    STRUCTURAL_HARVEST_TARGET_HZ,
    VALIDATION_MESH,
    harvest_ext_result_json,
    load_harvest_extension_manifest,
    run_mpi_harvest_solve,
    write_json,
)


def _sample_by_id(manifest: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    for s in manifest.get("samples") or []:
        if str(s.get("id")) == sample_id:
            return s
    raise KeyError(sample_id)


def run_harvest_solves(manifest: Dict[str, Any], *, resume: bool) -> List[Dict[str, Any]]:
    policy = manifest.get("harvest_policy") or {}
    band_lo = float(policy.get("harvest_lo_hz", STRUCTURAL_HARVEST_LO))
    band_hi = float(policy.get("harvest_hi_hz", STRUCTURAL_HARVEST_HI))
    target_hz = float(policy.get("shift_invert_target_hz", STRUCTURAL_HARVEST_TARGET_HZ))
    num_modes = int(policy.get("num_modes", STRUCTURAL_HARVEST_NUM_MODES))
    mesh = VALIDATION_MESH
    if not mesh.is_file():
        raise FileNotFoundError(f"validation mesh missing: {mesh}")

    HARVEST_EXT_SAMPLES.mkdir(parents=True, exist_ok=True)
    HARVEST_EXT_DIAG.mkdir(parents=True, exist_ok=True)
    case_reports: List[Dict[str, Any]] = []

    for sample_id in manifest.get("sample_ids") or []:
        sample = _sample_by_id(manifest, str(sample_id))
        sid = str(sample["id"])
        log_path = HARVEST_EXT_SAMPLES / sid / "logs" / "harvest_extension_solve.log"
        if resume and harvest_ext_result_json(sid):
            print(f"[harvest_ext] skip solve (resume): {sid}", flush=True)
            solve = json.loads(harvest_ext_result_json(sid).read_text(encoding="utf-8"))
            report = case_spectrum_report(sample, solve, band_lo=band_lo, band_hi=band_hi)
            case_reports.append(report)
            write_json(HARVEST_EXT_SAMPLES / sid / "diagnostics" / "harvest_case_report.json", report)
            continue

        print(
            f"[harvest_ext] solve {sid} band=[{band_lo},{band_hi}] target={target_hz} nm={num_modes}",
            flush=True,
        )
        rc, solve = run_mpi_harvest_solve(
            sample,
            mesh,
            target_hz=target_hz,
            harvest_lo_hz=band_lo,
            harvest_hi_hz=band_hi,
            num_modes=num_modes,
            log_path=log_path,
            case_root=HARVEST_EXT_SAMPLES,
        )
        report = case_spectrum_report(sample, solve, band_lo=band_lo, band_hi=band_hi)
        report["solve_exit_code"] = int(rc)
        case_reports.append(report)
        write_json(HARVEST_EXT_SAMPLES / sid / "diagnostics" / "harvest_case_report.json", report)
        if rc != 0 or not solve.get("v2_converged"):
            print(f"[harvest_ext] WARNING: {sid} rc={rc} v2_converged={solve.get('v2_converged')}", flush=True)

    write_json(
        HARVEST_EXT_DIAG / "harvest_solve_summary.json",
        {
            "suite": manifest.get("suite"),
            "harvest_policy": policy,
            "mesh_file": str(mesh),
            "case_spectrum_reports": case_reports,
        },
    )
    return case_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 material structural harvest extension")
    parser.add_argument("--skip-solve", action="store_true", help="Run post-analysis only")
    parser.add_argument("--post-only", action="store_true", help="Alias for --skip-solve")
    parser.add_argument("--resume", action="store_true", help="Skip solves when result JSON exists")
    parser.add_argument(
        "--apply-promotion",
        action="store_true",
        help="Forward to post step: promote structural gate if criterion met",
    )
    args = parser.parse_args()
    skip_solve = bool(args.skip_solve or args.post_only)

    manifest = load_harvest_extension_manifest()
    if not skip_solve:
        run_harvest_solves(manifest, resume=bool(args.resume))

    import subprocess

    cmd = [sys.executable, str(SCRIPT_DIR / "run_v2_material_structural_harvest_post.py")]
    if args.apply_promotion:
        cmd.append("--apply-promotion")
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
