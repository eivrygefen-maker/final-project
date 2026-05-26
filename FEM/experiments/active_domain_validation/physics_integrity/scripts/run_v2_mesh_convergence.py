#!/usr/bin/env python3
"""
Experiment-only v2_mesh_convergence: resumable multi-level mesh study (L0–L_check).

Frozen coupled_physical_core_v2; no LHS, no promotion, no artifact cleanup.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_mesh_convergence_common import (
    CONV_DIAG,
    CONV_MESH,
    CONV_ROOT,
    INCREMENTAL_JSON,
    VALIDATION_MESH,
    load_manifest,
    mesh_audit_path,
    run_mpi_case_solve,
    sample_spec_from_case,
    solve_case_dir,
    solve_done,
    solve_result_path,
    write_json,
)
from v2_mesh_convergence_mesh import build_level_mesh
from v2_sensitivity_gates import run_mesh_gates
from v2_sensitivity_mesh import sample_mesh_path

CONFIG_DIR = CONV_ROOT / "configs"


def _ingest_l0(case: Dict[str, Any], level_id: str) -> Optional[Dict[str, Any]]:
    reuse = case.get("l0_reuse") or {}
    if level_id != "L0" or not reuse.get("enabled"):
        return None
    rel = str(reuse.get("artifact_dir", ""))
    if not rel:
        return None
    art = (SCRIPT_DIR.parent / rel).resolve() if not Path(rel).is_absolute() else Path(rel)
    results_dir = art / "results"
    if not results_dir.is_dir():
        return None
    files = sorted(results_dir.glob("result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    solve = json.loads(files[0].read_text(encoding="utf-8"))
    solve["ingested_from"] = str(art)
    solve["ingested_l0_reuse"] = True
    return solve


def _resolve_l0_mesh(case: Dict[str, Any]) -> Path:
    cid = str(case["id"])
    if cid == "baseline_coupled_v2":
        return VALIDATION_MESH
    if cid == "hole_radius_large":
        p = sample_mesh_path(cid)
        if p.is_file():
            return p.resolve()
    return VALIDATION_MESH


def _write_incremental(state: Dict[str, Any]) -> None:
    write_json(INCREMENTAL_JSON, state)


def _run_one(
    manifest: Dict[str, Any],
    case: Dict[str, Any],
    level_id: str,
    level_def: Dict[str, Any],
    state: Dict[str, Any],
    *,
    resume: bool,
) -> None:
    cid = str(case["id"])
    key = f"{level_id}/{cid}"
    levels = manifest.get("mesh_levels") or {}
    if level_id == "L_check" and level_def.get("optional"):
        state.setdefault("L_check_attempted", True)

    case_dir = solve_case_dir(level_id, cid)
    target_hz = float(case["target_hz"])

    if resume and solve_done(level_id, case):
        print(f"[mesh_conv] skip solve (done): {key}", flush=True)
        return

    mesh_file: Optional[Path] = None
    mesh_meta: Dict[str, Any] = {}

    if level_id == "L0":
        mesh_file = _resolve_l0_mesh(case)
        mesh_meta = {"reused_validation_mesh": str(mesh_file), "built": False}
        ingested = None
        if cid != "material_back_cedar" and (case.get("l0_reuse") or {}).get("enabled"):
            ingested = _ingest_l0(case, level_id)
        if ingested:
            rp = solve_result_path(level_id, cid, target_hz)
            rp.parent.mkdir(parents=True, exist_ok=True)
            ingested["mesh_file"] = str(mesh_file)
            ingested["mesh_level"] = level_id
            rp.write_text(json.dumps(ingested, indent=2), encoding="utf-8")
            state.setdefault("completed", []).append(key)
            state.setdefault("rows", {})[key] = {
                "status": "ingested_l0",
                "case_type": case.get("case_type"),
                "mesh_meta": mesh_meta,
            }
            _write_incremental(state)
            print(f"[mesh_conv] ingested L0: {key}", flush=True)
            return
        if mesh_file and mesh_file.is_file() and not mesh_audit_path(level_id, cid).is_file():
            from v2_mesh_convergence_mesh import _mesh_audit

            audit = _mesh_audit(mesh_file, mesh_audit_path(level_id, cid))
            mesh_meta["mesh_audit"] = audit
    if level_id != "L0":
        try:
            mesh_meta = build_level_mesh(case, level_id, level_def, config_dir=CONFIG_DIR)
        except Exception as exc:
            state.setdefault("failures", []).append({key: str(exc)})
            if level_id == "L_check" and level_def.get("skip_on_resource_failure"):
                state["L_check_skipped"] = str(exc)
                _write_incremental(state)
                return
            raise
        if mesh_meta.get("build_failed"):
            msg = f"mesh build failed {key}"
            if level_id == "L_check" and level_def.get("skip_on_resource_failure"):
                state["L_check_skipped"] = msg
                state.setdefault("failures", []).append({key: msg})
                _write_incremental(state)
                return
            raise RuntimeError(msg)
        mesh_file = Path(mesh_meta["mesh_file"])

    if mesh_file is None or not mesh_file.is_file():
        raise FileNotFoundError(f"mesh missing for {key}: {mesh_file}")

    if level_def.get("run_gates_on_build") and mesh_file.is_file():
        gates = run_mesh_gates(
            mesh_file,
            hole_radius_m=float((case.get("geometry") or {}).get("hole_radius", 0.047)),
            gates_dir=case_dir / "diagnostics" / "gates",
        )
        mesh_meta["mesh_gates"] = gates
        if not gates.get("combined_mesh_gate_pass"):
            state.setdefault("failures", []).append({key: "mesh_gate_failed"})
            _write_incremental(state)
            raise RuntimeError(f"mesh gates failed for {key}")

    sample = sample_spec_from_case(case)
    log_path = case_dir / "logs" / "mesh_convergence_solve.log"
    t0 = time.perf_counter()
    rc, solve = run_mpi_case_solve(
        sample,
        mesh_file,
        target_hz=target_hz,
        harvest_lo_hz=float(case["harvest_lo_hz"]),
        harvest_hi_hz=float(case["harvest_hi_hz"]),
        num_modes=int(case["num_modes"]),
        log_path=log_path,
        case_dir=case_dir,
        select_by_energy=bool(case.get("select_by_energy")),
        structural_spectrum_harvest=bool(case.get("structural_spectrum_harvest")),
    )
    elapsed = time.perf_counter() - t0
    solve["mesh_level"] = level_id
    solve["mesh_audit"] = mesh_meta
    solve["solve_exit_code"] = rc
    solve["elapsed_s"] = elapsed
    rp = solve_result_path(level_id, cid, target_hz)
    write_json(rp, solve)

    status = "ok" if rc == 0 and solve.get("v2_converged") else "solve_failed"
    if status != "ok" and level_id == "L_check" and level_def.get("skip_on_resource_failure"):
        state["L_check_skipped"] = f"solve failed {key} rc={rc}"
        state.setdefault("failures", []).append({key: f"rc={rc}"})
        _write_incremental(state)
        return
    state.setdefault("completed", []).append(key)
    state.setdefault("rows", {})[key] = {"status": status, "elapsed_s": elapsed, "mesh_meta": mesh_meta}
    _write_incremental(state)
    print(f"[mesh_conv] solved {key} status={status} elapsed={elapsed:.1f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 mesh convergence stage")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--skip-solve", action="store_true", help="Post-process only")
    parser.add_argument("--skip-L-check", action="store_true", help="Do not attempt L_check level")
    args = parser.parse_args()

    manifest = load_manifest()
    CONV_ROOT.mkdir(parents=True, exist_ok=True)
    CONV_DIAG.mkdir(parents=True, exist_ok=True)
    CONV_MESH.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "suite": manifest.get("suite"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": [],
        "rows": {},
        "failures": [],
    }
    if INCREMENTAL_JSON.is_file() and args.resume:
        try:
            state.update(json.loads(INCREMENTAL_JSON.read_text(encoding="utf-8")))
        except Exception:
            pass

    levels_order = list(manifest.get("level_run_order") or ["L0", "L_mid", "L_prod", "L_check"])
    if args.skip_L_check:
        levels_order = [x for x in levels_order if x != "L_check"]

    if not args.skip_solve:
        for level_id in levels_order:
            level_def = (manifest.get("mesh_levels") or {}).get(level_id) or {}
            for case in manifest.get("cases") or []:
                if level_id == "L_check" and args.skip_L_check:
                    continue
                _run_one(manifest, case, level_id, level_def, state, resume=bool(args.resume))

    import subprocess

    cmd = [sys.executable, str(SCRIPT_DIR / "run_v2_mesh_convergence_post.py")]
    return int(subprocess.call(cmd))


if __name__ == "__main__":
    raise SystemExit(main())
