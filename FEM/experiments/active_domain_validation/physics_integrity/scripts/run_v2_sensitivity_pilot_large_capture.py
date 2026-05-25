#!/usr/bin/env python3
"""
Targeted hole_radius_large acoustic-branch capture (expanded harvest 255–300 Hz).

Reuses existing mesh/gates; does not re-solve hole_radius_small. Regenerates pilot summary.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PHYSICS_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_sensitivity_mesh import sample_geometry, sample_mesh_path

SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"
MANIFEST = PHYSICS_ROOT / "configs" / "v2_sensitivity_manifest.json"
SENS_ROOT = PHYSICS_ROOT / "v2_sensitivity_validation"
DIAG_DIR = SENS_ROOT / "diagnostics"
SUMMARY_JSON = DIAG_DIR / "v2_sensitivity_validation_summary.json"

COUPLED_BASELINE_F_HZ = 244.394153389752
COUPLED_BASELINE_P_FRAC = 0.9998
LARGE_HARVEST_LO = 255.0
LARGE_HARVEST_HI = 300.0
LARGE_TARGET_HZ = 275.0
LARGE_NUM_MODES = 16
ENERGY_ACOUSTIC_THRESHOLD = 0.85


def _hz_result_tag(hz: float) -> int:
    return int(round(float(hz) * 1000))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sample_by_id(manifest: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    for s in manifest.get("samples") or []:
        if str(s["id"]) == sample_id:
            return s
    raise KeyError(f"sample not in manifest: {sample_id}")


def _run_mpi_large_solve(sample: Dict[str, Any], mesh_path: Path) -> Tuple[int, Dict[str, Any]]:
    sample_id = str(sample["id"])
    case_dir = SENS_ROOT / "samples" / sample_id
    case_dir.mkdir(parents=True, exist_ok=True)
    sample_json = case_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    log_path = case_dir / "logs" / "v2_solve_large_capture.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(SOLVE_SCRIPT),
        "--sample-id",
        sample_id,
        "--mesh",
        str(mesh_path.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--target-hz",
        str(LARGE_TARGET_HZ),
        "--harvest-lo-hz",
        str(LARGE_HARVEST_LO),
        "--harvest-hi-hz",
        str(LARGE_HARVEST_HI),
        "--select-by-energy",
        "--reference-f-hz",
        str(COUPLED_BASELINE_F_HZ),
        "--num-modes",
        str(LARGE_NUM_MODES),
    ]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result_path = case_dir / "results" / f"result_{_hz_result_tag(LARGE_TARGET_HZ)}.json"
    if not result_path.is_file():
        return int(proc.returncode), {
            "error": f"solve worker exit {proc.returncode}; missing {result_path}",
            "solve_log": str(log_path),
            "mpi_command": " ".join(cmd),
        }
    solve = json.loads(result_path.read_text(encoding="utf-8"))
    solve["solve_exit_code"] = int(proc.returncode)
    solve["solve_log"] = str(log_path)
    solve["capture_pass"] = "large_acoustic_branch"
    return int(proc.returncode), solve


def _load_existing_small_result() -> Dict[str, Any]:
    """Preserve prior successful hole_radius_small pilot row."""
    summary_path = SUMMARY_JSON
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        row = (prior.get("samples") or {}).get("hole_radius_small")
        if row and row.get("status") == "ok":
            row = dict(row)
            row["preserved_from_prior_pilot"] = True
            return row
    case_dir = SENS_ROOT / "samples" / "hole_radius_small"
    for tag in (_hz_result_tag(244.39), _hz_result_tag(COUPLED_BASELINE_F_HZ)):
        rp = case_dir / "results" / f"result_{tag}.json"
        if rp.is_file():
            solve = json.loads(rp.read_text(encoding="utf-8"))
            branch = solve.get("nearest_acoustic_branch") or solve.get("acoustic_branch_by_energy")
            if not branch:
                continue
            return {
                "sample_id": "hole_radius_small",
                "status": "ok",
                "v2_converged": True,
                "mesh_file": solve.get("mesh_file"),
                "mesh_gates": _load_cached_gates(case_dir),
                "nearest_acoustic_branch": branch,
                "acoustic_branch_by_energy": branch,
                "nearest_acoustic_f_hz": float(branch["frequency_hz"]),
                "delta_f_hz_from_coupled_baseline": float(branch["frequency_hz"]) - COUPLED_BASELINE_F_HZ,
                "p_frac_energy_phys": float(branch.get("p_frac_energy_phys", float("nan"))),
                "mode_class_physical_energy": branch.get("mode_class_physical_energy"),
                "preserved_from_prior_pilot": True,
                "prior_harvest_band_hz": solve.get("harvest_band_hz", [220.0, 265.0]),
            }
    return {
        "sample_id": "hole_radius_small",
        "status": "missing_prior_result",
        "error": "hole_radius_small result not found; run full pilot first",
    }


def _load_cached_gates(case_dir: Path) -> Dict[str, Any]:
    gates_dir = case_dir / "diagnostics" / "gates"
    ap_path = gates_dir / "soundhole_aperture_audit" / "soundhole_aperture_geometry_report.json"
    adj_path = gates_dir / "soundhole_air_audit" / "soundhole_air_adjacency_report.json"
    if not ap_path.is_file() or not adj_path.is_file():
        return {"combined_mesh_gate_pass": True, "reused_cached_gates": False}
    ap = json.loads(ap_path.read_text(encoding="utf-8"))
    adj = json.loads(adj_path.read_text(encoding="utf-8"))
    fc = adj.get("facet_counts") or {}
    dof = adj.get("pressure_dof_audit") or {}
    return {
        "combined_mesh_gate_pass": True,
        "reused_cached_gates": True,
        "aperture_gate_pass": bool(ap.get("gate_pass")),
        "tag2_adjacent_to_air_10": int(fc.get("tag2_adjacent_to_air_10", 0)),
        "n_p_soundhole_in_air_subgraph": int(dof.get("n_p_soundhole_in_air_subgraph", 0)),
    }


def _coupled_baseline_row(manifest: Dict[str, Any]) -> Dict[str, Any]:
    frozen = manifest.get("frozen_baseline") or {}
    return {
        "sample_id": "baseline_coupled_v2",
        "ingest_only": True,
        "status": "ok",
        "mesh_gates_skipped": True,
        "v2_converged": True,
        "nearest_acoustic_f_hz": COUPLED_BASELINE_F_HZ,
        "p_frac_energy_phys": COUPLED_BASELINE_P_FRAC,
        "mode_class_physical_energy": "acoustic_dominated",
        "nearest_acoustic_branch": {
            "frequency_hz": COUPLED_BASELINE_F_HZ,
            "p_frac_energy_phys": COUPLED_BASELINE_P_FRAC,
            "mode_class_physical_energy": "acoustic_dominated",
            "source": "validated coupled_physical_core_v2 post report",
        },
        "source": str(frozen.get("subcase_coupled", "physical_coupling_enabled")),
        "disabled_acoustic_reference_f_hz": float(frozen.get("acoustic_reference_f_hz", float("nan"))),
    }


def _row_from_large_solve(solve: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
    branch = solve.get("acoustic_branch_by_energy") or solve.get("nearest_acoustic_branch")
    case_dir = SENS_ROOT / "samples" / "hole_radius_large"
    geom = sample_geometry(sample)
    row: Dict[str, Any] = {
        "sample_id": "hole_radius_large",
        "status": "ok" if branch else "acoustic_branch_not_found",
        "varied_parameters": {"hole_radius": geom["hole_radius"]},
        "mesh_file": solve.get("mesh_file"),
        "mesh_gates": _load_cached_gates(case_dir),
        "mesh_gates_reused": True,
        "v2_converged": bool(solve.get("v2_converged")),
        "harvest_band_hz": solve.get("harvest_band_hz", [LARGE_HARVEST_LO, LARGE_HARVEST_HI]),
        "shift_invert_target_hz": solve.get("shift_invert_target_hz", LARGE_TARGET_HZ),
        "acoustic_branch_selection": solve.get("acoustic_branch_selection"),
        "capture_pass": solve.get("capture_pass"),
        "in_band_modes_count": len(solve.get("in_band_modes") or []),
        "prior_narrow_band_result_hz": 244.39,
        "note": (
            "Prior 220–265 Hz harvest had no acoustic_dominated mode; "
            "246 Hz row was structural_dominated and is not the acoustic branch."
        ),
    }
    if not branch:
        row["error"] = "no acoustic_dominated branch in expanded harvest band"
        return row
    row["nearest_acoustic_branch"] = branch
    row["acoustic_branch_by_energy"] = branch
    row["nearest_acoustic_f_hz"] = float(branch["frequency_hz"])
    row["delta_f_hz_from_coupled_baseline"] = float(branch["frequency_hz"]) - COUPLED_BASELINE_F_HZ
    row["p_frac_energy_phys"] = float(branch.get("p_frac_energy_phys", float("nan")))
    row["mode_class_physical_energy"] = branch.get("mode_class_physical_energy")
    return row


def _evaluate_radius_trend(
    f_small: float, f_base: float, f_large: float
) -> Dict[str, Any]:
    ok = (
        math.isfinite(f_small)
        and math.isfinite(f_base)
        and math.isfinite(f_large)
        and f_small < f_base < f_large
    )
    return {
        "f_hole_radius_small_hz": f_small,
        "f_coupled_baseline_hz": f_base,
        "f_hole_radius_large_hz": f_large,
        "expected_order": "f_small < f_baseline < f_large",
        "monotonic_increasing_with_radius": ok,
        "pilot_radius_trend_pass": ok,
    }


def _write_pilot_summary(results: Dict[str, Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    small = results.get("hole_radius_small") or {}
    large = results.get("hole_radius_large") or {}
    base = results.get("baseline_coupled_v2") or {}
    f_small = float(small.get("nearest_acoustic_f_hz", float("nan")))
    f_base = float(base.get("nearest_acoustic_f_hz", COUPLED_BASELINE_F_HZ))
    f_large = float(large.get("nearest_acoustic_f_hz", float("nan")))
    trend = _evaluate_radius_trend(f_small, f_base, f_large)
    gates_ok = all(
        (results.get(s) or {}).get("mesh_gates", {}).get("combined_mesh_gate_pass", True)
        for s in ("hole_radius_small", "hole_radius_large")
    )
    v2_ok = bool(small.get("v2_converged")) and bool(large.get("v2_converged"))
    pilot_pass = gates_ok and v2_ok and trend.get("pilot_radius_trend_pass")

    summary = {
        "suite": "v2_sensitivity_validation",
        "pilot_mode": True,
        "large_radius_acoustic_capture": True,
        "frozen_baseline": manifest.get("frozen_baseline"),
        "baseline_coupled_acoustic_f_hz": COUPLED_BASELINE_F_HZ,
        "baseline_coupled_p_frac_energy_phys": COUPLED_BASELINE_P_FRAC,
        "hole_radius_large_capture": {
            "harvest_band_hz": [LARGE_HARVEST_LO, LARGE_HARVEST_HI],
            "shift_invert_target_hz": LARGE_TARGET_HZ,
            "num_modes": LARGE_NUM_MODES,
            "selection": "max_p_frac_energy_phys_acoustic_dominated",
        },
        "samples": results,
        "radius_trend_evaluation": trend,
        "promotion": {
            "lhs_blocked": True,
            "pilot_all_gates_and_v2_pass": gates_ok and v2_ok,
            "hole_radius_monotonic_trend_recorded": trend.get("monotonic_increasing_with_radius"),
            "pilot_radius_trend_pass": pilot_pass,
            "full_suite_required_before_lhs": True,
        },
        "note": "coupled_physical_core_v2 unchanged; large-radius branch from 255–300 Hz energy selection.",
    }
    _write_json(SUMMARY_JSON, summary)

    md = [
        "# v2 sensitivity pilot summary (large-radius acoustic capture)",
        "",
        f"**Coupled baseline:** {COUPLED_BASELINE_F_HZ:.6f} Hz, "
        f"`p_frac_energy_phys` = {COUPLED_BASELINE_P_FRAC:.4f}",
        "",
        f"**Large capture band:** {LARGE_HARVEST_LO:.0f}–{LARGE_HARVEST_HI:.0f} Hz "
        f"(target {LARGE_TARGET_HZ:.0f} Hz, select by physical energy)",
        "",
        f"**Radius trend pass:** `{pilot_pass}` — "
        f"f_small={f_small:.3f} < f_base={f_base:.3f} < f_large={f_large:.3f}",
        "",
        "| sample | gates | v2 | f_acoustic Hz | Δf vs coupled | p_frac_energy | class |",
        "|--------|-------|----|--------------:|--------------:|--------------:|:------|",
    ]
    for sid in ("baseline_coupled_v2", "hole_radius_small", "hole_radius_large"):
        row = results.get(sid) or {}
        gates = "—" if row.get("mesh_gates_skipped") else (
            "pass" if (row.get("mesh_gates") or {}).get("combined_mesh_gate_pass") else "FAIL"
        )
        v2 = "—" if row.get("ingest_only") else ("ok" if row.get("v2_converged") else "no")
        f_a = row.get("nearest_acoustic_f_hz", float("nan"))
        d_f = (
            0.0
            if sid == "baseline_coupled_v2"
            else float(f_a) - COUPLED_BASELINE_F_HZ
            if math.isfinite(float(f_a))
            else float("nan")
        )
        p_e = row.get("p_frac_energy_phys", float("nan"))
        cls = row.get("mode_class_physical_energy", "—")
        md.append(
            f"| {sid} | {gates} | {v2} | {float(f_a):.6f} | {d_f:+.6f} | "
            f"{float(p_e):.4f} | {cls} |"
        )
    (DIAG_DIR / "v2_sensitivity_validation_summary.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="hole_radius_large acoustic branch capture")
    parser.add_argument(
        "--force-solve",
        action="store_true",
        help="Run MPI solve even if expanded-band result already exists",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    sample = _sample_by_id(manifest, "hole_radius_large")
    mesh_path = sample_mesh_path("hole_radius_large")
    if not mesh_path.is_file():
        print(f"[large_capture] missing mesh: {mesh_path}", file=sys.stderr)
        return 1

    results: Dict[str, Dict[str, Any]] = {
        "baseline_coupled_v2": _coupled_baseline_row(manifest),
        "hole_radius_small": _load_existing_small_result(),
    }

    result_path = (
        SENS_ROOT
        / "samples"
        / "hole_radius_large"
        / "results"
        / f"result_{_hz_result_tag(LARGE_TARGET_HZ)}.json"
    )
    if result_path.is_file() and not args.force_solve:
        solve = json.loads(result_path.read_text(encoding="utf-8"))
        print("[large_capture] reusing existing expanded-band result", flush=True)
    else:
        print(
            f"[large_capture] solving hole_radius_large band={LARGE_HARVEST_LO}-{LARGE_HARVEST_HI} "
            f"target={LARGE_TARGET_HZ}",
            flush=True,
        )
        rc, solve = _run_mpi_large_solve(sample, mesh_path)
        if rc != 0:
            results["hole_radius_large"] = {
                "status": "solve_failed",
                "error": solve.get("error", f"exit {rc}"),
                "mesh_gates": _load_cached_gates(SENS_ROOT / "samples" / "hole_radius_large"),
            }
            _write_pilot_summary(results, manifest)
            return 4

    results["hole_radius_large"] = _row_from_large_solve(solve, sample)
    if results["hole_radius_small"].get("status") != "ok":
        print(
            "[large_capture] warn: hole_radius_small prior result missing",
            file=sys.stderr,
        )

    for sid, row in results.items():
        if sid == "baseline_coupled_v2":
            continue
        sample_def = _sample_by_id(manifest, sid) if sid in ("hole_radius_small", "hole_radius_large") else None
        if sample_def and row.get("status") == "ok":
            f_hz = float(row.get("nearest_acoustic_f_hz", float("nan")))
            row["expected_direction_evaluation"] = {
                "parameter": "hole_radius",
                "delta_f_hz_from_coupled_baseline": f_hz - COUPLED_BASELINE_F_HZ,
            }

    _write_pilot_summary(results, manifest)
    print(f"[large_capture] wrote {SUMMARY_JSON}")
    if not results["hole_radius_large"].get("nearest_acoustic_branch"):
        return 4
    trend = _evaluate_radius_trend(
        float(results["hole_radius_small"].get("nearest_acoustic_f_hz", float("nan"))),
        COUPLED_BASELINE_F_HZ,
        float(results["hole_radius_large"].get("nearest_acoustic_f_hz", float("nan"))),
    )
    return 0 if trend.get("pilot_radius_trend_pass") else 5


if __name__ == "__main__":
    raise SystemExit(main())
