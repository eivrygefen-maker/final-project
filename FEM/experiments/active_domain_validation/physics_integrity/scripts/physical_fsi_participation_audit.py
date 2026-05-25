#!/usr/bin/env python3
"""
No-eigensolve audit: physical-FSI-only mode at ~245.30 Hz vs decoupled-union acoustic at ~244.39 Hz.

Assembles reduced-domain A/M once (physical FSI, Nitsche off), replays saved mode vectors,
computes participation metrics and pressure-subspace MAC, and refreshes experiment verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fem_main_3d as fem3d
from fem_worker_single import hz_result_tag

from coupled_participation_audit import (
    _acoustic_reference_hz,
    _load_coupled_mode_dense_vector,
    _load_modes,
    _resolve_mesh,
    _write_json,
)
from mode_diagnostics import (
    P_FRAC_FULLY_UNSCALED_DEFINITION,
    P_FRAC_PHYS_GNHEP_DEFINITION,
    P_FRAC_PRODUCTION_DEFINITION,
    block_l2_p_fraction,
    compute_mass_energy_participation,
    diagnose_mixed_mode,
    evaluate_physical_fsi_acoustic_survival,
    merge_scaling_metadata,
    pressure_subspace_mac,
    unscale_mixed_mode_vector,
)

DECOUPLED_CASE = PHYSICS_ROOT / "coupled_decoupled_union"
PHYSICAL_CASE = PHYSICS_ROOT / "coupled_physical_fsi_only"
DECOUPLED_CONFIG = PHYSICS_ROOT / "configs" / "coupled_decoupled_union.json"
PHYSICAL_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_fsi_only.json"

WOOD_PARTICIPATION_NOTE = (
    "wood_participation in mode_physics_diagnostics is top_plate_frac + back_plate_frac from "
    "harvest/sifter: each term is the fraction of structural (shell) kinetic energy on that plate "
    "facet tag relative to the total structural energy on top+back, not air-vs-wood volume "
    "participation. With tiny ||u|| and p-dominated modes it often saturates at 1.0 because "
    "essentially all resolved structural energy sits on one plate tag; it does not contradict "
    "pressure dominance in the global mixed eigenvector."
)

MAC_SCALING_NOTE = (
    "Pressure MAC uses only active air-supported pressure DOFs (n_p_active collapse indices mapped "
    "to reduced W). Three scalings are reported: (1) raw — SLEPc/GNHEP-assembled coefficients; "
    "(2) gnhep_undo — multiply p by s_pp (undo block Frobenius only); (3) fully_unscaled — "
    "multiply p by s_pp/pressure_dof_scale. Phase alignment uses |dot| so sign flips are ignored."
)


def _find_mode_near(
    modes_meta: List[Dict[str, Any]],
    mode_files: List[Path],
    *,
    target_hz: float,
    tol_hz: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], float]:
    best_meta: Optional[Dict[str, Any]] = None
    best_path: Optional[Path] = None
    best_f = float("nan")
    best_d = tol_hz
    for path in mode_files:
        try:
            mode_index = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        meta = next((m for m in modes_meta if int(m.get("mode_index", -1)) == mode_index), {})
        f_hz = float(meta.get("frequency_hz", float("nan")))
        if not np.isfinite(f_hz):
            continue
        d = abs(f_hz - target_hz)
        if d <= best_d:
            best_d = d
            best_f = f_hz
            best_meta = meta
            best_path = path
    return best_meta, best_path, best_f


def _audit_candidate_mode(
    vec: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
    M: Any,
    A: Any,
    mode_meta: Dict[str, Any],
    frequency_hz: float,
) -> Dict[str, Any]:
    diag = diagnose_mixed_mode(
        vec,
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        gnhep=gnhep,
        wood_top=float(mode_meta.get("top_plate_frac", 0.0)),
        wood_back=float(mode_meta.get("back_plate_frac", 0.0)),
        frequency_hz=frequency_hz,
    )
    p_prod, u_norm, p_norm = block_l2_p_fraction(vec, u_to_W=u_to_W, p_to_W=p_to_W)
    vec_unscaled = unscale_mixed_mode_vector(
        vec, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep, undo_pressure_dof_scale=True
    )
    p_full, _, _ = block_l2_p_fraction(vec_unscaled, u_to_W=u_to_W, p_to_W=p_to_W)
    energy = compute_mass_energy_participation(
        vec, M, A, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep
    )
    return {
        **diag,
        "p_frac_production": float(
            mode_meta.get("p_frac_production", mode_meta.get("p_frac_raw", p_prod))
        ),
        "p_frac_raw_audit": float(p_prod),
        "p_frac_fully_unscaled": float(p_full),
        "u_norm_solver_coords": float(u_norm),
        "p_norm_solver_coords": float(p_norm),
        "structural_modal_energy_phys": float(energy["structural_modal_energy_phys"]),
        "acoustic_modal_energy_phys": float(energy["acoustic_modal_energy_phys"]),
        "p_frac_energy_phys": float(energy["p_frac_energy_phys"]),
        "mass_cross_term_phys": float(energy["mass_cross_term_phys"]),
        "mass_cross_u_from_p_gnhep": float(energy["mass_cross_u_from_p_gnhep"]),
        "mass_cross_p_from_u_gnhep": float(energy["mass_cross_p_from_u_gnhep"]),
    }


def _pressure_mac_report(
    vec_ref: np.ndarray,
    vec_cand: np.ndarray,
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
) -> Dict[str, Any]:
    s_p = max(float(gnhep.get("s_pp", 1.0)), 1.0e-30)
    p_scale = max(float(gnhep.get("pressure_dof_scale", 30.0)), 1.0e-30)
    return {
        "mac_pressure_raw": pressure_subspace_mac(vec_ref, vec_cand, p_to_W),
        "mac_pressure_gnhep_undo_s_pp": pressure_subspace_mac(
            vec_ref, vec_cand, p_to_W, scale_p_a=s_p, scale_p_b=s_p
        ),
        "mac_pressure_fully_unscaled": pressure_subspace_mac(
            vec_ref,
            vec_cand,
            p_to_W,
            scale_p_a=s_p / p_scale,
            scale_p_b=s_p / p_scale,
        ),
        "scaling_documentation": MAC_SCALING_NOTE,
        "n_p_active_dofs": int(np.asarray(p_to_W, dtype=np.int32).size),
    }


def _refresh_physical_fsi_summary(
    *,
    audit: Dict[str, Any],
    ref_hz: float,
    freq_tol_hz: float,
) -> None:
    cand = audit["physical_fsi_candidate"]
    mac = float(cand["pressure_overlap"]["mac_pressure_gnhep_undo_s_pp"])
    verdict, survives, note, detail = evaluate_physical_fsi_acoustic_survival(
        frequency_hz=float(cand["frequency_hz"]),
        ref_hz=ref_hz,
        freq_tol_hz=freq_tol_hz,
        p_frac_production=float(cand["p_frac_production"]),
        p_frac_phys_gnhep=float(cand["p_frac_phys_gnhep"]),
        p_frac_fully_unscaled=float(cand["p_frac_fully_unscaled"]),
        p_frac_energy_phys=float(cand["p_frac_energy_phys"]),
        pressure_mac_gnhep_undo=mac,
    )
    summary_path = PHYSICAL_CASE / "diagnostics" / "physical_fsi_only_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "verdict": verdict,
            "acoustic_branch_survives": survives,
            "verdict_note": note,
            "verdict_criteria": detail,
            "post_audit_refreshed": True,
            "participation_audit_json": "physical_fsi_participation_audit.json",
        }
    )
    _write_json(summary_path, summary)
    md = PHYSICAL_CASE / "diagnostics" / "physical_fsi_only_summary.md"
    md.write_text(
        "\n".join(
            [
                "# Physical-FSI-only isolation (Nitsche disabled)",
                "",
                f"**Verdict (post-audit):** `{verdict}`",
                "",
                f"Note: {note}",
                "",
                f"Candidate: f={cand['frequency_hz']:.4f} Hz, "
                f"MAC(s_pp)={mac:.4f}, p_frac_production={cand['p_frac_production']:.4e}, "
                f"p_frac_energy_phys={cand['p_frac_energy_phys']:.4e}",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if MPI.COMM_WORLD.rank == 0:
        print(f"[physical_fsi_participation_audit] refreshed summary verdict: {verdict}")
        print(f"[physical_fsi_participation_audit] {note}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Physical-FSI participation / overlap audit")
    parser.add_argument("--physical-freq-hz", type=float, default=245.299844)
    parser.add_argument("--decoupled-freq-hz", type=float, default=244.3916)
    parser.add_argument("--freq-match-tol-hz", type=float, default=0.15)
    parser.add_argument("--mac-threshold", type=float, default=0.85)
    parser.add_argument("--skip-summary-refresh", action="store_true")
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[physical_fsi_participation_audit] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    physical_cfg = json.loads(PHYSICAL_CONFIG.read_text(encoding="utf-8"))
    solver_cfg = physical_cfg.setdefault("solver", {})
    target_hz = float(solver_cfg.get("_worker_target_hz", 244.39))
    ref_hz = _acoustic_reference_hz(solver_cfg)
    freq_tol = float(solver_cfg.get("physical_fsi_acoustic_tol_hz", 1.0))

    sorting = PHYSICAL_CASE / "sorting_physical_fsi_audit"
    sorting.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())
    solver_cfg["physics_integrity_capture"] = True
    solver_cfg["active_domain_experiment"] = {"enabled": False}
    solver_cfg["coupled_air_pressure_restriction_replay_audit"] = True

    mesh_file = _resolve_mesh(physical_cfg, PHYSICAL_CONFIG)
    if MPI.COMM_WORLD.rank == 0:
        print(
            "[physical_fsi_participation_audit] Assembling physical-FSI-only A/M "
            "(reduced domain, no SLEPc)...",
            flush=True,
        )
    _msh, _W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_file,
        config=physical_cfg,
        num_modes=0,
        solve_evp=False,
    )
    u_to_W = np.asarray(physical_cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(physical_cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    n_W = int(A.getSize()[0])
    restr = physical_cfg.get("_coupled_air_pressure_restriction") or {}

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[physical_fsi_participation_audit] reduced W={n_W} "
            f"n_u_active={restr.get('n_u_active')} n_p_active={restr.get('n_p_active')}",
            flush=True,
        )

    gnhep = merge_scaling_metadata(PHYSICAL_CASE)
    pi = physical_cfg.get("_physics_integrity") or {}
    if isinstance(pi, dict) and pi.get("gnhep_scales"):
        gnhep.update({k: float(v) for k, v in pi["gnhep_scales"].items()})

    hz_tag = hz_result_tag(target_hz)
    phys_meta, phys_files = _load_modes(PHYSICAL_CASE, target_hz)
    dec_meta, dec_files = _load_modes(DECOUPLED_CASE, target_hz)
    if not phys_files or not dec_files:
        print("[physical_fsi_participation_audit] Missing saved mode files", file=sys.stderr)
        return 1

    cand_meta, cand_path, cand_f = _find_mode_near(
        phys_meta, phys_files, target_hz=args.physical_freq_hz, tol_hz=args.freq_match_tol_hz
    )
    ref_meta, ref_path, ref_f = _find_mode_near(
        dec_meta, dec_files, target_hz=args.decoupled_freq_hz, tol_hz=args.freq_match_tol_hz
    )
    if cand_path is None or ref_path is None:
        print(
            "[physical_fsi_participation_audit] Could not match modes within "
            f"±{args.freq_match_tol_hz} Hz",
            file=sys.stderr,
        )
        return 1

    vec_cand, cand_load = _load_coupled_mode_dense_vector(
        cand_path, n_coupled_W=n_W, mode_index=int(cand_meta.get("mode_index", 0))
    )
    vec_ref, ref_load = _load_coupled_mode_dense_vector(
        ref_path, n_coupled_W=n_W, mode_index=int(ref_meta.get("mode_index", 0))
    )

    cand_audit = _audit_candidate_mode(
        vec_cand,
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        gnhep=gnhep,
        M=M,
        A=A,
        mode_meta=cand_meta or {},
        frequency_hz=cand_f,
    )
    ref_audit = _audit_candidate_mode(
        vec_ref,
        u_to_W=u_to_W,
        p_to_W=p_to_W,
        gnhep=gnhep,
        M=M,
        A=A,
        mode_meta=ref_meta or {},
        frequency_hz=ref_f,
    )
    mac_report = _pressure_mac_report(vec_ref, vec_cand, p_to_W, gnhep)
    cand_audit["pressure_overlap_vs_decoupled_acoustic"] = mac_report
    cand_audit["delta_from_decoupled_mode_hz"] = float(cand_f - ref_f)
    cand_audit["mode_load"] = cand_load
    ref_audit["mode_load"] = ref_load
    ref_audit["role"] = "decoupled_union_acoustic_reference"

    verdict, survives, note, detail = evaluate_physical_fsi_acoustic_survival(
        frequency_hz=cand_f,
        ref_hz=ref_hz,
        freq_tol_hz=freq_tol,
        p_frac_production=float(cand_audit["p_frac_production"]),
        p_frac_phys_gnhep=float(cand_audit["p_frac_phys_gnhep"]),
        p_frac_fully_unscaled=float(cand_audit["p_frac_fully_unscaled"]),
        p_frac_energy_phys=float(cand_audit["p_frac_energy_phys"]),
        pressure_mac_gnhep_undo=float(mac_report["mac_pressure_gnhep_undo_s_pp"]),
        mac_threshold=float(args.mac_threshold),
    )

    report: Dict[str, Any] = {
        "experiment": "physical_fsi_participation_overlap_audit",
        "no_eigensolve": True,
        "acoustic_reference_hz": ref_hz,
        "decoupled_union_reference_mode": {
            "frequency_hz": ref_f,
            "vector_path": str(ref_path.relative_to(DECOUPLED_CASE)).replace("\\", "/"),
            **ref_audit,
        },
        "physical_fsi_candidate": {
            "frequency_hz": cand_f,
            "vector_path": str(cand_path.relative_to(PHYSICAL_CASE)).replace("\\", "/"),
            **cand_audit,
        },
        "pressure_overlap": mac_report,
        "wood_participation_interpretation": WOOD_PARTICIPATION_NOTE,
        "metric_definitions": {
            "p_frac_production": P_FRAC_PRODUCTION_DEFINITION,
            "p_frac_phys_gnhep": P_FRAC_PHYS_GNHEP_DEFINITION,
            "p_frac_fully_unscaled": P_FRAC_FULLY_UNSCALED_DEFINITION,
            "p_frac_energy_phys": (
                "E_air_phys / (E_struct + E_air + |cross|) from x^T M x on assembled "
                "physical-FSI-only reduced operator."
            ),
        },
        "scaling_metadata": gnhep,
        "reduced_domain": restr,
        "verdict_recommended": verdict,
        "acoustic_branch_survives_recommended": survives,
        "verdict_note": note,
        "verdict_criteria": detail,
        "metric_conflict_explanation": (
            "p_frac_phys_gnhep divides GNHEP-undone ||p|| by (||u||_gnhep_undo + ||p||_gnhep_undo). "
            "When ||u|| is tiny in solver coordinates but FSI coupling inflates the u-block after "
            "s_uu scaling, the denominator is u-dominated and p_frac_phys_gnhep can be O(1e-4) "
            "even with production p_frac≈1 and ||p||≈1. Energy participation and pressure MAC "
            "against the decoupled acoustic mode are the appropriate branch-survival indicators."
        ),
    }

    diag_dir = PHYSICAL_CASE / "diagnostics"
    out_json = diag_dir / "physical_fsi_participation_audit.json"
    out_md = diag_dir / "physical_fsi_participation_audit.md"
    _write_json(out_json, report)

    if MPI.COMM_WORLD.rank == 0:
        lines = [
            "# Physical-FSI participation / overlap audit",
            "",
            f"**Recommended verdict:** `{verdict}`",
            "",
            f"{note}",
            "",
            "## Candidate mode (physical FSI only)",
            "",
            f"- f = **{cand_f:.6f} Hz** (Δ decoupled mode = {cand_f - ref_f:+.4f} Hz, "
            f"Δ acoustic ref = {cand_f - ref_hz:+.4f} Hz)",
            f"- p_frac production = {cand_audit['p_frac_production']:.6e}",
            f"- p_frac GNHEP undo = {cand_audit['p_frac_phys_gnhep']:.6e}",
            f"- p_frac fully unscaled = {cand_audit['p_frac_fully_unscaled']:.6e}",
            f"- p_frac energy (phys) = {cand_audit['p_frac_energy_phys']:.6e}",
            f"- E_air phys = {cand_audit['acoustic_modal_energy_phys']:.6e}",
            f"- E_struct phys = {cand_audit['structural_modal_energy_phys']:.6e}",
            f"- mass cross (phys) = {cand_audit['mass_cross_term_phys']:.6e}",
            f"- ||u||, ||p|| (solver) = {cand_audit['u_norm_solver_coords']:.6e}, "
            f"{cand_audit['p_norm_solver_coords']:.6e}",
            "",
            "## Pressure overlap vs decoupled acoustic (active p only)",
            "",
            f"- MAC raw = {mac_report['mac_pressure_raw']:.6f}",
            f"- MAC gnhep_undo (s_pp) = {mac_report['mac_pressure_gnhep_undo_s_pp']:.6f}",
            f"- MAC fully unscaled = {mac_report['mac_pressure_fully_unscaled']:.6f}",
            "",
            MAC_SCALING_NOTE,
            "",
            "## Wood participation",
            "",
            WOOD_PARTICIPATION_NOTE,
            "",
            "## Metric conflict",
            "",
            report["metric_conflict_explanation"],
            "",
        ]
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[physical_fsi_participation_audit] candidate f={cand_f:.6f} Hz")
        print(
            f"  p_frac prod={cand_audit['p_frac_production']:.4e} "
            f"gnhep_undo={cand_audit['p_frac_phys_gnhep']:.4e} "
            f"energy={cand_audit['p_frac_energy_phys']:.4e}"
        )
        print(
            f"  MAC(s_pp)={mac_report['mac_pressure_gnhep_undo_s_pp']:.4f} "
            f"recommended={verdict}"
        )
        print(f"[physical_fsi_participation_audit] wrote {out_json} and {out_md}")

    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    if not args.skip_summary_refresh:
        _refresh_physical_fsi_summary(audit=report, ref_hz=ref_hz, freq_tol_hz=freq_tol)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
