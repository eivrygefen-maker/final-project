#!/usr/bin/env python3
"""
Post-process physical-FSI alpha continuation (pilot: reuse alpha=0/1 endpoints + one solve).

No eigensolve for endpoints; optional A/M replay for energy metrics on pilot alpha only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from fem_worker_single import hz_result_tag

from coupled_participation_audit import (
    _acoustic_reference_hz,
    _catalog_saved_modes,
    _load_coupled_mode_dense_vector,
    _load_modes,
    _resolve_mesh,
    _write_json,
)
from mode_diagnostics import (
    block_l2_p_fraction,
    compute_mass_energy_participation,
    merge_scaling_metadata,
    pressure_subspace_mac,
)

DECOUPLED_CASE = PHYSICS_ROOT / "coupled_decoupled_union"
PHYSICAL_FSI_CASE = PHYSICS_ROOT / "coupled_physical_fsi_only"
PILOT_CASE = PHYSICS_ROOT / "coupled_physical_fsi_alpha_pilot"
PILOT_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_fsi_alpha_pilot.json"

ALPHA0_HZ = 244.3916
FULL_ALPHA_SEQUENCE = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0)
PILOT_ALPHA_SEQUENCE = (0.0, 0.01, 1.0)
BAND_LO = 220.0
BAND_HI = 265.0
SIGMA_SUSPECT_HZ = 273.7168
SIGMA_SUSPECT_TOL_HZ = 0.05
MAC_HIGH_THRESHOLD = 0.85
ENERGY_DOMINANCE_THRESHOLD = 0.35
SELF_MATCH_TOL_HZ = 0.15


def _case_for_alpha(alpha: float, *, pilot: bool) -> Tuple[float, Path, bool]:
    """Return (alpha, case_dir, requires_solve)."""
    if abs(alpha) <= 1.0e-15:
        return 0.0, DECOUPLED_CASE, False
    if abs(alpha - 1.0) <= 1.0e-15:
        return 1.0, PHYSICAL_FSI_CASE, False
    if pilot and abs(alpha - 0.01) <= 1.0e-9:
        return 0.01, PILOT_CASE, True
    raise ValueError(f"No case directory for alpha={alpha} (pilot={pilot})")


def _select_reference_mode(
    catalog: List[Dict[str, Any]],
    *,
    target_hz: float,
    prefer_acoustic: bool,
) -> Optional[Dict[str, Any]]:
    pool = list(catalog)
    if prefer_acoustic:
        acoustic = [
            e
            for e in catalog
            if e.get("mode_class") == "acoustic_dominated"
            or float(e.get("p_frac_production", 0.0)) >= ENERGY_DOMINANCE_THRESHOLD
        ]
        if acoustic:
            pool = acoustic
    hits = [
        e for e in pool if abs(float(e["frequency_hz"]) - target_hz) <= SELF_MATCH_TOL_HZ
    ]
    if hits:
        return min(hits, key=lambda e: abs(float(e["frequency_hz"]) - target_hz))
    return min(pool, key=lambda e: abs(float(e["frequency_hz"]) - target_hz), default=None)


def _pressure_dominated_in_band(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for e in catalog:
        f_hz = float(e["frequency_hz"])
        if not (BAND_LO <= f_hz <= BAND_HI):
            continue
        if (
            e.get("mode_class") == "acoustic_dominated"
            or float(e.get("p_frac_production", 0.0)) >= ENERGY_DOMINANCE_THRESHOLD
            or float(e.get("p_frac_phys_gnhep", 0.0)) >= ENERGY_DOMINANCE_THRESHOLD
        ):
            rows.append(e)
    return sorted(rows, key=lambda r: float(r["frequency_hz"]))


def _mac_pair(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
) -> Dict[str, float]:
    s_p = max(float(gnhep.get("s_pp", 1.0)), 1.0e-30)
    p_scale = max(float(gnhep.get("pressure_dof_scale", 30.0)), 1.0e-30)
    return {
        "mac_raw": pressure_subspace_mac(vec_a, vec_b, p_to_W),
        "mac_gnhep_undo_s_pp": pressure_subspace_mac(
            vec_a, vec_b, p_to_W, scale_p_a=s_p, scale_p_b=s_p
        ),
        "mac_fully_unscaled": pressure_subspace_mac(
            vec_a,
            vec_b,
            p_to_W,
            scale_p_a=s_p / p_scale,
            scale_p_b=s_p / p_scale,
        ),
    }


def _meta_energy_frac(case_dir: Path, mode_index: int) -> Optional[float]:
    path = case_dir / "diagnostics" / "mode_physics_diagnostics.json"
    if not path.is_file():
        return None
    for m in json.loads(path.read_text(encoding="utf-8")).get("modes") or []:
        if int(m.get("mode_index", -1)) == mode_index:
            return float(m.get("p_frac_energy_phys", m.get("p_frac_phys_gnhep", float("nan"))))
    return None


def _analyze_alpha_step(
    *,
    alpha: float,
    case_dir: Path,
    catalog: List[Dict[str, Any]],
    vec_by_index: Dict[int, np.ndarray],
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
    ref_hz: float,
    alpha0_hz: float,
    prev_selected: Optional[Dict[str, Any]],
    alpha0_vec: np.ndarray,
    M: Any,
    A: Any,
    u_to_W: np.ndarray,
    energy_from_operator: bool,
) -> Dict[str, Any]:
    competitors = _pressure_dominated_in_band(catalog)
    ref_mode = _select_reference_mode(catalog, target_hz=ALPHA0_HZ, prefer_acoustic=True)
    if ref_mode is None and competitors:
        ref_mode = competitors[0]

    best_mac_prev = -1.0
    best_row: Optional[Dict[str, Any]] = None
    competitor_rows: List[Dict[str, Any]] = []

    for entry in competitors:
        mi = int(entry["mode_index"])
        vec = vec_by_index[mi]
        f_hz = float(entry["frequency_hz"])
        p_prod, _, _ = block_l2_p_fraction(vec, u_to_W=u_to_W, p_to_W=p_to_W)
        if energy_from_operator and M is not None and A is not None:
            energy = compute_mass_energy_participation(
                vec, M, A, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep
            )
            p_energy = float(energy["p_frac_energy_phys"])
            e_air = float(energy["acoustic_modal_energy_phys"])
        else:
            p_energy = float(_meta_energy_frac(case_dir, mi) or float("nan"))
            e_air = float("nan")
        mac_ref0 = _mac_pair(vec, alpha0_vec, p_to_W, gnhep)
        mac_prev: Dict[str, float] = {}
        if prev_selected is not None:
            mac_prev = _mac_pair(
                vec,
                prev_selected["vector"],
                p_to_W,
                gnhep,
            )
        sigma_suspect = abs(f_hz - SIGMA_SUSPECT_HZ) <= SIGMA_SUSPECT_TOL_HZ
        row = {
            "mode_index": mi,
            "frequency_hz": f_hz,
            "vector_path": entry["vector_path"],
            "branch_shift_hz": f_hz - alpha0_hz,
            "delta_from_acoustic_reference_hz": f_hz - ref_hz,
            "p_frac_production": float(
                entry.get("p_frac_production", p_prod)
            ),
            "p_frac_energy_phys": p_energy,
            "acoustic_modal_energy_phys": e_air,
            "energy_metric_source": (
                "assembled_M_at_pilot_alpha"
                if energy_from_operator
                else "saved_mode_physics_diagnostics"
            ),
            "sigma_mapped_suspect": sigma_suspect,
            "mac_to_alpha0_reference": mac_ref0,
            "mac_to_previous_selected": mac_prev if prev_selected else None,
            "mode_class": entry.get("mode_class"),
        }
        competitor_rows.append(row)
        mac_track = float(
            mac_prev.get("mac_gnhep_undo_s_pp", -1.0)
            if prev_selected
            else mac_ref0.get("mac_gnhep_undo_s_pp", -1.0)
        )
        if mac_track > best_mac_prev:
            best_mac_prev = mac_track
            best_row = {**row, "vector": vec}

    selected = best_row or (competitor_rows[0] if competitor_rows else None)
    if selected is None:
        return {
            "alpha_fsi": alpha,
            "case_dir": str(case_dir),
            "error": "no pressure-dominated modes in band",
        }

    return {
        "alpha_fsi": alpha,
        "case_dir": str(case_dir.relative_to(PHYSICS_ROOT)).replace("\\", "/"),
        "selected_mode": {
            k: v for k, v in selected.items() if k != "vector"
        },
        "competing_pressure_dominated_modes": competitor_rows,
        "selection_rule": (
            "maximize mac_gnhep_undo_s_pp vs previous selected mode"
            if prev_selected
            else "maximize mac_gnhep_undo_s_pp vs alpha=0 reference at first step"
        ),
    }


def _pilot_verdict(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_alpha = {float(s["alpha_fsi"]): s for s in steps}
    s001 = by_alpha.get(0.01)
    s000 = by_alpha.get(0.0)
    if not s001 or s001.get("error"):
        return {"outcome": "PILOT_INCOMPLETE", "note": "alpha=0.01 step missing"}

    sel = s001.get("selected_mode") or {}
    mac_prev = (sel.get("mac_to_previous_selected") or {}).get("mac_gnhep_undo_s_pp")
    mac_ref0 = (sel.get("mac_to_alpha0_reference") or {}).get("mac_gnhep_undo_s_pp")
    f_hz = float(sel.get("frequency_hz", float("nan")))

    if mac_prev is not None and float(mac_prev) < 0.15 and abs(f_hz - ALPHA0_HZ) > 2.0:
        return {
            "outcome": "PHYSICAL_FSI_ASSEMBLY_SUSPECT_AT_SMALL_ALPHA",
            "alpha_fsi": 0.01,
            "mac_to_previous": mac_prev,
            "mac_to_alpha0": mac_ref0,
            "note": "MAC collapsed at alpha=0.01 with large frequency jump",
        }
    if mac_prev is not None and float(mac_prev) >= MAC_HIGH_THRESHOLD:
        return {
            "outcome": "PHYSICAL_FSI_BRANCH_CONTINUOUS",
            "note": f"Pilot: consecutive MAC(0→0.01)={mac_prev:.4f} >= {MAC_HIGH_THRESHOLD}",
        }
    if mac_prev is not None:
        return {
            "outcome": "PHYSICAL_FSI_BRANCH_BREAKS_AT_ALPHA",
            "alpha_fsi": 0.01,
            "mac_to_previous": mac_prev,
            "mac_to_alpha0": mac_ref0,
            "note": "Branch shape not continuous from decoupled acoustic at alpha=0.01",
        }
    return {"outcome": "PILOT_INCONCLUSIVE", "note": "MAC metrics unavailable"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical-FSI continuation post-process")
    parser.add_argument("--pilot", action="store_true", help="Use pilot alpha sequence 0,0.01,1")
    parser.add_argument("--skip-am-replay", action="store_true", help="Skip A/M assembly for energy")
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[physical_fsi_continuation_post] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    alphas = list(PILOT_ALPHA_SEQUENCE if args.pilot else FULL_ALPHA_SEQUENCE)
    ref_hz = _acoustic_reference_hz(
        json.loads(PILOT_CONFIG.read_text(encoding="utf-8")).get("solver", {})
    )
    target_hz = 244.39

    p_to_W: Optional[np.ndarray] = None
    u_to_W: Optional[np.ndarray] = None
    gnhep = merge_scaling_metadata(PHYSICAL_FSI_CASE)
    M = A = None
    n_W = 0

    if not args.skip_am_replay:
        cfg = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
        cfg.setdefault("solver", {})["coupled_air_pressure_restriction_replay_audit"] = True
        sorting = PILOT_CASE / "sorting_continuation_post"
        sorting.mkdir(parents=True, exist_ok=True)
        fem3d.set_sorting_root(sorting.resolve())
        mesh = _resolve_mesh(cfg, PILOT_CONFIG)
        if MPI.COMM_WORLD.rank == 0:
            print(
                "[physical_fsi_continuation_post] A/M replay at pilot alpha=0.01 for energy metrics",
                flush=True,
            )
        _msh, _W, A, M = fem3d._solve_coupled_evp(
            mesh_file=mesh,
            config=cfg,
            num_modes=0,
            solve_evp=False,
        )
        p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
        u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
        n_W = int(A.getSize()[0])

    steps: List[Dict[str, Any]] = []
    prev_selected: Optional[Dict[str, Any]] = None
    alpha0_vec: Optional[np.ndarray] = None

    for alpha in alphas:
        case_dir = _case_for_alpha(alpha, pilot=args.pilot)[1]
        meta, files = _load_modes(case_dir, target_hz)
        catalog = _catalog_saved_modes(case_dir, meta, files, target_hz=target_hz)
        if not catalog:
            steps.append(
                {"alpha_fsi": alpha, "case_dir": str(case_dir), "error": "empty catalog"}
            )
            continue

        vec_by_index: Dict[int, np.ndarray] = {}
        for entry in catalog:
            if p_to_W is None:
                gnhep = merge_scaling_metadata(case_dir)
            n_case = n_W
            if n_case == 0:
                n_case = 112100
            vec, _ = _load_coupled_mode_dense_vector(
                entry["path"], n_coupled_W=n_case, mode_index=int(entry["mode_index"])
            )
            vec_by_index[int(entry["mode_index"])] = vec

        if abs(alpha) <= 1.0e-15:
            ref_entry = _select_reference_mode(
                catalog, target_hz=ALPHA0_HZ, prefer_acoustic=True
            )
            if ref_entry is None:
                steps.append({"alpha_fsi": 0.0, "error": "no alpha=0 reference mode"})
                continue
            alpha0_vec = vec_by_index[int(ref_entry["mode_index"])]
            alpha0_hz = float(ref_entry["frequency_hz"])
            prev_selected = {
                "vector": alpha0_vec,
                "frequency_hz": alpha0_hz,
                "mode_index": int(ref_entry["mode_index"]),
            }
            steps.append(
                {
                    "alpha_fsi": 0.0,
                    "case_dir": str(case_dir.relative_to(PHYSICS_ROOT)).replace("\\", "/"),
                    "reused_endpoint": True,
                    "selected_mode": {
                        "mode_index": int(ref_entry["mode_index"]),
                        "frequency_hz": alpha0_hz,
                        "vector_path": ref_entry["vector_path"],
                        "branch_shift_hz": 0.0,
                        "p_frac_production": float(ref_entry.get("p_frac_production", 0.0)),
                        "note": "decoupled-union acoustic reference (244.3916 Hz)",
                    },
                }
            )
            continue

        if alpha0_vec is None:
            steps.append({"alpha_fsi": alpha, "error": "alpha=0 reference not loaded"})
            continue

        if p_to_W is None:
            steps.append({"alpha_fsi": alpha, "error": "p_to_W map unavailable"})
            continue
        step = _analyze_alpha_step(
            alpha=alpha,
            case_dir=case_dir,
            catalog=catalog,
            vec_by_index=vec_by_index,
            p_to_W=p_to_W,
            gnhep=gnhep,
            ref_hz=ref_hz,
            alpha0_hz=float(prev_selected["frequency_hz"]),
            prev_selected=prev_selected,
            alpha0_vec=alpha0_vec,
            M=M,
            A=A,
            u_to_W=u_to_W,
            energy_from_operator=use_op_energy,
        )
        step["reused_endpoint"] = abs(alpha - 1.0) <= 1.0e-15
        sel = step.get("selected_mode")
        if sel and int(sel.get("mode_index", -1)) in vec_by_index:
            prev_selected = {
                "vector": vec_by_index[int(sel["mode_index"])],
                "frequency_hz": float(sel["frequency_hz"]),
                "mode_index": int(sel["mode_index"]),
            }
        steps.append(step)

    verdict = _pilot_verdict(steps) if args.pilot else {
        "outcome": "FULL_SWEEP_NOT_RUN",
        "note": "Prepare interpretation from per-alpha steps; run full sweep after pilot approval.",
    }

    report = {
        "experiment": "physical_fsi_continuation_pilot" if args.pilot else "physical_fsi_continuation",
        "alpha_sequence": alphas,
        "full_alpha_sequence_prepared": list(FULL_ALPHA_SEQUENCE),
        "acoustic_reference_hz": ref_hz,
        "alpha0_reference_hz": ALPHA0_HZ,
        "band_hz": [BAND_LO, BAND_HI],
        "mac_high_threshold": MAC_HIGH_THRESHOLD,
        "continuation_steps": steps,
        "diagnostic_outcome": verdict,
        "next_actions": {
            "if_PHYSICAL_FSI_BRANCH_CONTINUOUS": "Consider full alpha sweep",
            "if_BRANCH_BREAKS_OR_ASSEMBLY_SUSPECT": (
                "Physical FSI coupling-strength / sign audit before Nitsche isolation"
            ),
            "do_not_run_yet": ["nitsche_isolation", "production_gain_tuning"],
        },
    }

    out_dir = PILOT_CASE if args.pilot else PHYSICS_ROOT / "coupled_physical_fsi_continuation"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = out_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    _write_json(diag / "physical_fsi_continuation_report.json", report)

    if MPI.COMM_WORLD.rank == 0:
        md_lines = [
            "# Physical-FSI continuation (pilot)" if args.pilot else "# Physical-FSI continuation",
            "",
            f"**Outcome:** `{verdict.get('outcome')}`",
            "",
            verdict.get("note", ""),
            "",
            f"Alpha sequence: {alphas}",
            "",
            "## Per-alpha steps",
            "",
        ]
        for step in steps:
            alpha = step.get("alpha_fsi")
            sel = step.get("selected_mode") or {}
            md_lines.append(f"### alpha_fsi = {alpha}")
            if step.get("error"):
                md_lines.append(f"- ERROR: {step['error']}")
                continue
            md_lines.append(
                f"- f={sel.get('frequency_hz', '?'):} Hz branch_shift={sel.get('branch_shift_hz', 0):+.4f}"
            )
            m0 = sel.get("mac_to_alpha0_reference") or {}
            mp = sel.get("mac_to_previous_selected") or {}
            md_lines.append(
                f"- MAC to alpha=0: raw={m0.get('mac_raw', float('nan')):.4f} "
                f"s_pp={m0.get('mac_gnhep_undo_s_pp', float('nan')):.4f}"
            )
            if mp:
                md_lines.append(
                    f"- MAC to previous: raw={mp.get('mac_raw', float('nan')):.4f} "
                    f"s_pp={mp.get('mac_gnhep_undo_s_pp', float('nan')):.4f}"
                )
            md_lines.append(
                f"- p_frac_energy={sel.get('p_frac_energy_phys', float('nan')):.4e} "
                f"p_frac_prod={sel.get('p_frac_production', float('nan')):.4e}"
            )
            md_lines.append("")
        (diag / "physical_fsi_continuation_report.md").write_text(
            "\n".join(md_lines) + "\n", encoding="utf-8"
        )
        print(f"[physical_fsi_continuation_post] outcome={verdict.get('outcome')}")
        print(f"[physical_fsi_continuation_post] wrote {diag / 'physical_fsi_continuation_report.json'}")

    try:
        if M is not None:
            M.destroy()
        if A is not None:
            A.destroy()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
