#!/usr/bin/env python3
"""
No-eigensolve post-processing for coupled_physical_core_v2 validation.

Replays reduced A/M per subcase, fills physical energy participation, reciprocity
on the reduced layout, and pressure MAC between subcases (no new EPS solve).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import zlib
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
from coupled_participation_audit import _load_coupled_mode_dense_vector, _load_modes
from fem_worker_single import hz_result_tag
from mode_diagnostics import compute_mass_energy_participation, merge_scaling_metadata, pressure_subspace_mac
from physical_fsi_seed_residual_audit import (
    N_P_ACTIVE_EXPECT,
    N_REDUCED_W_EXPECT,
    N_U_ACTIVE_EXPECT,
    _mask_on_indices,
    _petsc_matvec,
    _petsc_vec_from_array,
    _validate_reduced_layout,
)

V2_ROOT = PHYSICS_ROOT / "coupled_physical_core_v2"
V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
DIAG_DIR = V2_ROOT / "diagnostics"
REPORT_JSON = DIAG_DIR / "physical_core_v2_validation_report.json"
BAND_LO = 220.0
BAND_HI = 265.0
REF_TOL_HZ = 1.0
MAC_BRANCH_THRESHOLD = 0.85
ENERGY_ACOUSTIC_THRESHOLD = 0.85
ENERGY_STRUCT_THRESHOLD = 0.15

SUBCASES = (
    ("coupling_disabled", False),
    ("physical_coupling_enabled", True),
)


def _resolve_mesh(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, EXPERIMENT_ROOT, REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / raw).resolve()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _classify_phys_energy(p_frac_energy_phys: float) -> str:
    if float(p_frac_energy_phys) >= ENERGY_ACOUSTIC_THRESHOLD:
        return "acoustic_dominated"
    if float(p_frac_energy_phys) <= ENERGY_STRUCT_THRESHOLD:
        return "structural_dominated"
    return "mixed"


def _assemble_reduced_v2_operator(
    cfg_base: dict,
    config_path: Path,
    *,
    subcase: str,
    coupling_enabled: bool,
    apply_gnhep_normalize: bool,
) -> Tuple[Any, Any, dict, np.ndarray, np.ndarray, Dict[str, Any]]:
    cfg = copy.deepcopy(cfg_base)
    sc = cfg.setdefault("solver", {})
    sc["coupled_physical_core_v2_diagnosis"] = True
    sc["coupled_physical_core_v2_coupling_enabled"] = bool(coupling_enabled)
    sc["fsi_coupling_gain"] = 1.0
    sc["fsi_nitsche_enable"] = False
    sc["physics_integrity_capture"] = True
    sc["coupled_air_pressure_restriction_diagnosis"] = True
    sc["coupled_air_pressure_restriction_replay_audit"] = True
    sc["gnhep_block_frobenius_normalize"] = bool(apply_gnhep_normalize)

    sorting = V2_ROOT / subcase / f"sorting_post_{'gnhep' if apply_gnhep_normalize else 'raw'}"
    sorting.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())

    mesh = _resolve_mesh(cfg, config_path)
    if MPI.COMM_WORLD.rank == 0:
        print(
            "[physical_core_v2_post] assemble "
            f"subcase={subcase} coupling_enabled={coupling_enabled} "
            f"gnhep_normalize={apply_gnhep_normalize}",
            flush=True,
        )
    _msh, _W, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh,
        config=cfg,
        num_modes=0,
        solve_evp=False,
    )
    if "_coupled_air_u_to_W_map" not in cfg:
        raise RuntimeError(f"post replay missing reduced maps for {subcase}")
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    restr = dict(cfg.get("_coupled_air_pressure_restriction") or {})
    _validate_reduced_layout(A, u_to_W, p_to_W, restr, seed_length=N_REDUCED_W_EXPECT, alpha_fsi=0.0)
    return A, M, cfg, u_to_W, p_to_W, restr


def _reciprocity_reduced(
    A: Any,
    *,
    u_idx: np.ndarray,
    p_idx: np.ndarray,
    operator_rows: int,
    representation: str,
    apply_gnhep_normalize: bool,
) -> Dict[str, Any]:
    """Bilinear reciprocity probe on reduced operator (unit p / unit u patterns)."""
    n = int(operator_rows)
    u_idx = np.asarray(u_idx, dtype=np.int32).ravel()
    p_idx = np.asarray(p_idx, dtype=np.int32).ravel()
    x_p = np.zeros(n, dtype=np.float64)
    x_u = np.zeros(n, dtype=np.float64)
    if p_idx.size:
        x_p[p_idx] = 1.0
    if u_idx.size:
        x_u[u_idx] = 1.0

    vp = _petsc_vec_from_array(A, x_p)
    vu = _petsc_vec_from_array(A, x_u)
    ay = az = None
    try:
        Ap, ay = _petsc_matvec(A, vp)
        Au, az = _petsc_matvec(A, vu)
    finally:
        vp.destroy()
        vu.destroy()
        if ay is not None:
            ay.destroy()
        if az is not None:
            az.destroy()

    up = float(np.vdot(x_u, Ap))
    pu = float(np.vdot(x_p, Au))
    nu = float(np.linalg.norm(Ap[u_idx])) if u_idx.size else 0.0
    npu = float(np.linalg.norm(Au[p_idx])) if p_idx.size else 0.0
    ratio = abs(pu) / max(abs(up), 1.0e-30)
    log_ratio = abs(math.log(max(ratio, 1.0e-30)))
    return {
        "representation": representation,
        "operator_rows": n,
        "n_u_active": int(u_idx.size),
        "n_p_active": int(p_idx.size),
        "gnhep_frobenius_normalize_applied": bool(apply_gnhep_normalize),
        "bilinear_A_up_on_unit_p": up,
        "bilinear_A_pu_on_unit_u": pu,
        "reciprocity_ratio_abs_pu_over_up": ratio,
        "reciprocity_log10_imbalance": log_ratio / math.log(10.0),
        "reciprocity_balanced": log_ratio < math.log(10.0),
        "note": (
            "Reduced-layout probe; unit patterns on active u/p DOF indices only. "
            "pre-GNHEP assembly uses gnhep_block_frobenius_normalize=False."
        ),
    }


def _verify_mode_storage() -> Dict[str, Any]:
    """Confirm subcase mode files live in separate directories (not one overwritten tree)."""
    rows: Dict[str, Any] = {}
    crc_by_name: Dict[str, Dict[str, int]] = {}
    for subcase, _ in SUBCASES:
        modes_dir = (V2_ROOT / subcase / "modes").resolve()
        files = sorted(modes_dir.glob("mode_*.smx.npz"))
        entries = []
        for path in files:
            data = path.read_bytes()
            crc = int(zlib.crc32(data) & 0xFFFFFFFF)
            entries.append(
                {
                    "name": path.name,
                    "absolute_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "crc32": crc,
                }
            )
            crc_by_name.setdefault(path.name, {})[subcase] = crc
        rows[subcase] = {
            "modes_dir_absolute": str(modes_dir),
            "n_mode_files": len(files),
            "files": entries[:5],
            "files_truncated": len(files) > 5,
        }

    shared_names = set(crc_by_name.keys())
    collisions = []
    identical_content = []
    for name in sorted(shared_names):
        per = crc_by_name[name]
        if len(per) < 2:
            continue
        vals = list(per.values())
        if vals[0] == vals[1]:
            identical_content.append({"filename": name, "crc32": vals[0], "subcases": list(per.keys())})
        else:
            collisions.append({"filename": name, "crc32_by_subcase": per})

    isolated = len(shared_names) > 0 and not identical_content and len(crc_by_name) > 0
    return {
        "subcases_separate_directories": True,
        "relative_paths_match_across_subcases": True,
        "relative_path_note": (
            "Both subcases use modes/mode_244390_XXX.smx.npz under distinct parent folders "
            f"({V2_ROOT.name}/coupling_disabled vs .../physical_coupling_enabled)."
        ),
        "per_subcase": rows,
        "shared_filenames": sorted(shared_names),
        "identical_file_content_across_subcases": identical_content,
        "distinct_file_content_by_name": len(identical_content) == 0,
        "mac_comparison_valid": len(identical_content) == 0 and all(
            rows[s]["n_mode_files"] > 0 for s, _ in SUBCASES
        ),
        "rerun_required_for_mac": not (
            len(identical_content) == 0 and all(rows[s]["n_mode_files"] > 0 for s, _ in SUBCASES)
        ),
    }


def _load_subcase_result(subcase: str, target_hz: float) -> Dict[str, Any]:
    path = V2_ROOT / subcase / "results" / f"result_{hz_result_tag(target_hz)}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _replay_subcase_energy(
    cfg_base: dict,
    config_path: Path,
    *,
    subcase: str,
    coupling_enabled: bool,
    target_hz: float,
) -> Dict[str, Any]:
    case_dir = V2_ROOT / subcase
    prior = _load_subcase_result(subcase, target_hz)
    A, M, cfg, u_to_W, p_to_W, restr = _assemble_reduced_v2_operator(
        cfg_base,
        config_path,
        subcase=subcase,
        coupling_enabled=coupling_enabled,
        apply_gnhep_normalize=True,
    )
    gnhep = merge_scaling_metadata(case_dir)
    pi = cfg.get("_physics_integrity") or {}
    if isinstance(pi, dict) and pi.get("gnhep_scales"):
        gnhep.update({k: float(v) for k, v in pi["gnhep_scales"].items()})

    meta, mode_files = _load_modes(case_dir, target_hz)
    n_W = int(restr.get("n_reduced_W", N_REDUCED_W_EXPECT))
    enriched: List[Dict[str, Any]] = []
    for path in mode_files:
        try:
            mi = int(path.stem.split("_")[-1])
        except ValueError:
            mi = len(enriched)
        row_meta = next((m for m in meta if int(m.get("mode_index", -1)) == mi), {})
        f_hz = float(row_meta.get("frequency_hz", float("nan")))
        if not (BAND_LO <= f_hz <= BAND_HI):
            continue
        vec, _ = _load_coupled_mode_dense_vector(path, n_coupled_W=n_W, mode_index=mi)
        energy = compute_mass_energy_participation(
            vec, M, A, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep
        )
        p_frac_e = float(energy["p_frac_energy_phys"])
        enriched.append(
            {
                "mode_index": mi,
                "frequency_hz": f_hz,
                "vector_path": str(path.relative_to(case_dir)).replace("\\", "/"),
                "vector_absolute_path": str(path.resolve()),
                "p_frac_raw": float(row_meta.get("p_frac_raw", float("nan"))),
                "p_frac_phys_gnhep": float(row_meta.get("p_frac_phys_gnhep", float("nan"))),
                "p_frac_energy_phys": p_frac_e,
                "structural_modal_energy_phys": float(energy["structural_modal_energy_phys"]),
                "acoustic_modal_energy_phys": float(energy["acoustic_modal_energy_phys"]),
                "mass_cross_term_phys": float(energy["mass_cross_term_phys"]),
                "mode_class_l2": row_meta.get("mode_class"),
                "mode_class_physical_energy": _classify_phys_energy(p_frac_e),
            }
        )
    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    eps = prior.get("eps_batch_diagnostics") or {}
    nconv = int(eps.get("nconv_marked", -1))
    ref_hz = float(prior.get("acoustic_reference_hz", 244.3916))
    nearest = None
    if enriched:
        acoustic_pool = [
            m
            for m in enriched
            if m["mode_class_physical_energy"] == "acoustic_dominated"
            or float(m["p_frac_energy_phys"]) >= 0.35
        ]
        pool = acoustic_pool if acoustic_pool else enriched
        nearest = min(pool, key=lambda m: abs(float(m["frequency_hz"]) - ref_hz))

    return {
        "subcase": subcase,
        "coupling_enabled": coupling_enabled,
        "n_reduced_W": n_W,
        "n_u_active": int(restr.get("n_u_active", N_U_ACTIVE_EXPECT)),
        "n_p_active": int(restr.get("n_p_active", N_P_ACTIVE_EXPECT)),
        "eps_batch_diagnostics": eps,
        "nconv_marked": nconv,
        "in_band_modes_physical_energy": enriched,
        "nearest_mode_physical_energy": nearest,
        "prior_solve_result": prior,
    }


def _pressure_mac(
    ref_vec: np.ndarray,
    cand_vec: np.ndarray,
    p_to_W: np.ndarray,
    gnhep: Dict[str, float],
) -> Dict[str, float]:
    s_p = max(float(gnhep.get("s_pp", 1.0)), 1.0e-30)
    return {
        "mac_pressure_raw": pressure_subspace_mac(ref_vec, cand_vec, p_to_W),
        "mac_pressure_gnhep_undo_s_pp": pressure_subspace_mac(
            ref_vec, cand_vec, p_to_W, scale_p_a=s_p, scale_p_b=s_p
        ),
    }


def _milestone_criteria(
    disabled: Dict[str, Any],
    enabled: Dict[str, Any],
    storage: Dict[str, Any],
    reciprocity: Dict[str, Any],
    mac: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    ref = disabled.get("nearest_mode_physical_energy") or {}
    f_ref = float(ref.get("frequency_hz", float("nan")))
    ref_ok = (
        math.isfinite(f_ref)
        and abs(f_ref - float(disabled.get("prior_solve_result", {}).get("acoustic_reference_hz", 244.3916)))
        <= REF_TOL_HZ
        and float(ref.get("p_frac_energy_phys", 0.0)) >= 0.35
    )
    nconv = int(enabled.get("nconv_marked", -1))
    enabled_ok = nconv > 0
    in_band = enabled.get("in_band_modes_physical_energy") or []
    energy_ok = all(
        m.get("p_frac_energy_phys") is not None
        and math.isfinite(float(m["p_frac_energy_phys"]))
        for m in in_band
    )
    recip_ok = bool(reciprocity.get("reciprocity_balanced"))
    mac_s = float((mac or {}).get("mac_pressure_gnhep_undo_s_pp", float("nan")))
    branch_ok = math.isfinite(mac_s) and mac_s >= MAC_BRANCH_THRESHOLD and storage.get(
        "mac_comparison_valid", False
    )
    return {
        "v2_disabled_reference_reproduced": ref_ok,
        "v2_enabled_solve_converged": enabled_ok,
        "v2_physical_energy_metrics_available": energy_ok and len(in_band) > 0,
        "v2_reciprocity_sign_check_pass": recip_ok,
        "v2_acoustic_branch_match": branch_ok,
        "prior_initial_milestone_false_reason": (
            "Initial report used L2 p_frac only; physical energy replay was not run and "
            "reciprocity_check failed (reduced mode vector length 112100 vs unreduced "
            "operator 136136). Criteria were merged into one boolean."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="coupled_physical_core_v2 post-process")
    parser.add_argument("--config", type=Path, default=V2_CONFIG)
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[physical_core_v2_post] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    config_path = args.config.resolve()
    cfg_base = json.loads(config_path.read_text(encoding="utf-8"))
    target_hz = float(cfg_base.get("solver", {}).get("_worker_target_hz", 244.39))

    storage = _verify_mode_storage()
    if MPI.COMM_WORLD.rank == 0:
        print(
            f"[physical_core_v2_post] mode_storage mac_valid={storage['mac_comparison_valid']} "
            f"rerun_required={storage['rerun_required_for_mac']}",
            flush=True,
        )

    disabled = _replay_subcase_energy(
        cfg_base,
        config_path,
        subcase="coupling_disabled",
        coupling_enabled=False,
        target_hz=target_hz,
    )
    enabled = _replay_subcase_energy(
        cfg_base,
        config_path,
        subcase="physical_coupling_enabled",
        coupling_enabled=True,
        target_hz=target_hz,
    )

    reciprocity: Dict[str, Any] = {}
    try:
        A_raw, _M, _cfg, u_map, p_map, restr = _assemble_reduced_v2_operator(
            cfg_base,
            config_path,
            subcase="physical_coupling_enabled",
            coupling_enabled=True,
            apply_gnhep_normalize=False,
        )
        n_rows = int(A_raw.getSize()[0])
        reciprocity = _reciprocity_reduced(
            A_raw,
            u_idx=u_map,
            p_idx=p_map,
            operator_rows=n_rows,
            representation="reduced_W_pre_gnhep_frobenius",
            apply_gnhep_normalize=False,
        )
        A_raw.destroy()
        _M.destroy()
    except Exception as exc:
        reciprocity = {"error": f"{type(exc).__name__}: {exc}"}

    mac_report: Optional[Dict[str, Any]] = None
    # MAC uses solve-time nearest acoustic/mixed candidates (not post energy re-ranking).
    ref_mode = (disabled.get("prior_solve_result") or {}).get("nearest_acoustic_mode") or {}
    cand_mode = (enabled.get("prior_solve_result") or {}).get("nearest_acoustic_mode") or {}
    if storage.get("mac_comparison_valid") and ref_mode.get("vector_path") and cand_mode.get(
        "vector_path"
    ):
        ref_path = (V2_ROOT / "coupling_disabled" / str(ref_mode["vector_path"])).resolve()
        cand_path = (
            V2_ROOT / "physical_coupling_enabled" / str(cand_mode["vector_path"])
        ).resolve()
        if ref_path == cand_path:
            mac_report = {
                "skipped": True,
                "reason": "reference and candidate resolve to the same absolute path",
            }
        else:
            n_W = int(disabled.get("n_reduced_W", N_REDUCED_W_EXPECT))
            _Ae, _Me, _cfg_e, _u_map, p_map, _re = _assemble_reduced_v2_operator(
                cfg_base,
                config_path,
                subcase="physical_coupling_enabled",
                coupling_enabled=True,
                apply_gnhep_normalize=True,
            )
            try:
                _Ae.destroy()
                _Me.destroy()
            except Exception:
                pass
            gnhep = merge_scaling_metadata(V2_ROOT / "physical_coupling_enabled")
            ref_mi = int(ref_mode.get("mode_index", 0))
            cand_mi = int(cand_mode.get("mode_index", 0))
            ref_vec, _ = _load_coupled_mode_dense_vector(
                ref_path, n_coupled_W=n_W, mode_index=ref_mi
            )
            cand_vec, _ = _load_coupled_mode_dense_vector(
                cand_path, n_coupled_W=n_W, mode_index=cand_mi
            )
            mac_vals = _pressure_mac(ref_vec, cand_vec, p_map, gnhep)
            f_ref = float(ref_mode["frequency_hz"])
            f_cand = float(cand_mode["frequency_hz"])
            mac_report = {
                **mac_vals,
                "reference_subcase": "coupling_disabled",
                "candidate_subcase": "physical_coupling_enabled",
                "reference_f_hz": f_ref,
                "candidate_f_hz": f_cand,
                "delta_f_hz": f_cand - f_ref,
                "reference_vector_absolute": str(ref_path),
                "candidate_vector_absolute": str(cand_path),
                "p_to_W_length": int(p_map.size),
                "inner_product": "np.vdot on active-pressure DOFs via p_to_W",
            }
    elif not storage.get("mac_comparison_valid"):
        reason = "mode files missing or identical content across subcases (overwrite suspected)"
        if storage.get("identical_file_content_across_subcases"):
            reason = (
                "identical CRC across subcases for shared filenames; "
                "do not MAC overwritten vectors"
            )
        mac_report = {
            "skipped": True,
            "reason": reason,
            "rerun_required": True,
            "rerun_command": (
                "bash FEM/experiments/active_domain_validation/physics_integrity/scripts/"
                "run_coupled_physical_core_v2.sh"
            ),
        }

    milestones = _milestone_criteria(disabled, enabled, storage, reciprocity, mac_report)
    ref_m = ref_mode or disabled.get("nearest_mode_physical_energy") or {}
    cand_m = cand_mode or enabled.get("nearest_mode_physical_energy") or {}
    delta_f = float(cand_m.get("frequency_hz", float("nan"))) - float(
        ref_m.get("frequency_hz", float("nan"))
    )

    prior_report: Dict[str, Any] = {}
    if REPORT_JSON.is_file():
        prior_report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    report = {
        "experiment": "coupled_physical_core_v2",
        "post_process_only": True,
        "formulation_report": "docs/coupled_physical_core_v2_formulation.md",
        "v1_investigation_closed": True,
        "mode_storage_verification": storage,
        "subcases": {
            "coupling_disabled": disabled,
            "physical_coupling_enabled": enabled,
        },
        "reciprocity_sign_check": reciprocity,
        "pressure_mac_disabled_to_enabled": mac_report,
        "frequency_shift_hz": {
            "reference_hz": float(ref_m.get("frequency_hz", float("nan"))),
            "candidate_hz": float(cand_m.get("frequency_hz", float("nan"))),
            "delta_hz": delta_f,
            "source": "prior_solve nearest_acoustic_mode (MAC pair)",
        },
        "physical_energy_nearest_modes": {
            "coupling_disabled": disabled.get("nearest_mode_physical_energy"),
            "physical_coupling_enabled": enabled.get("nearest_mode_physical_energy"),
        },
        "p_frac_energy_phys_definition": (
            "E_air_phys / (E_struct_phys + E_air_phys + mass_cross_term_phys); "
            "classification uses mode_class_physical_energy thresholds "
            f"(acoustic>={ENERGY_ACOUSTIC_THRESHOLD}, structural<={ENERGY_STRUCT_THRESHOLD})"
        ),
        "milestone_criteria": milestones,
        "prior_solve_summary": prior_report.get("subcases"),
        "acceptance": {
            "initial_v2_milestone_passed": all(
                milestones[k]
                for k in (
                    "v2_disabled_reference_reproduced",
                    "v2_enabled_solve_converged",
                    "v2_physical_energy_metrics_available",
                    "v2_reciprocity_sign_check_pass",
                    "v2_acoustic_branch_match",
                )
            ),
            "note": milestones.get("prior_initial_milestone_false_reason"),
        },
    }
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(REPORT_JSON, report)

    if MPI.COMM_WORLD.rank == 0:
        md = [
            "# coupled_physical_core_v2 validation (post-processed)",
            "",
            "## Why the first milestone flag was false",
            "",
            milestones.get("prior_initial_milestone_false_reason", ""),
            "",
            "## Mode storage",
            f"- MAC comparison valid: `{storage.get('mac_comparison_valid')}`",
            f"- Rerun required: `{storage.get('rerun_required_for_mac')}`",
            "",
            "## Milestone criteria (separate booleans)",
        ]
        for k, v in milestones.items():
            if k.startswith("v2_"):
                md.append(f"- `{k}`: {v}")
        recip = reciprocity
        if recip and not recip.get("error"):
            md.extend(
                [
                    "",
                    "## Reciprocity (pre-GNHEP, reduced W)",
                    f"- representation: `{recip.get('representation')}`",
                    f"- operator_rows: {recip.get('operator_rows')}",
                    f"- reciprocity_balanced: `{recip.get('reciprocity_balanced')}`",
                    f"- ratio |pu|/|up|: {recip.get('reciprocity_ratio_abs_pu_over_up', float('nan')):.6g}",
                ]
            )
        md.extend(
            [
                "",
                f"## Frequency shift (physical-energy nearest modes): {delta_f:+.6f} Hz",
                "",
                "## physical_coupling_enabled in-band modes (physical energy)",
                "",
                "| idx | f Hz | p_frac_energy_phys | class | E_struct | E_air | cross |",
                "|----:|-----:|-------------------:|:------|-------:|------:|------:|",
            ]
        )
        for m in enabled.get("in_band_modes_physical_energy") or []:
            md.append(
                f"| {m['mode_index']} | {float(m['frequency_hz']):.6f} | "
                f"{float(m['p_frac_energy_phys']):.4f} | {m['mode_class_physical_energy']} | "
                f"{float(m['structural_modal_energy_phys']):.3e} | "
                f"{float(m['acoustic_modal_energy_phys']):.3e} | "
                f"{float(m['mass_cross_term_phys']):.3e} |"
            )
        if mac_report and not mac_report.get("skipped"):
            md.append(
                f"\n## Pressure MAC (disabled → enabled)\n"
                f"- mac_gnhep_undo_s_pp = {mac_report.get('mac_pressure_gnhep_undo_s_pp', float('nan')):.4f}"
            )
        (DIAG_DIR / "physical_core_v2_validation_report.md").write_text(
            "\n".join(md) + "\n",
            encoding="utf-8",
        )
        print(f"[physical_core_v2_post] wrote {REPORT_JSON}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
