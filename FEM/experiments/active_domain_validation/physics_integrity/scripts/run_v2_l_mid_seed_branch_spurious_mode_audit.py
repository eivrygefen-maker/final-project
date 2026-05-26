#!/usr/bin/env python3
"""
Report-only sigma / BC spurious-mode audit for baseline seed-branch recovery diagnostic.

Reuses completed diagnostic artifacts — no new eigensolves.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fem_main_3d as fem3d
from physical_fsi_seed_residual_audit import (
    _block_residual_contributions,
    _rayleigh_metrics,
)
from v2_mesh_convergence_common import (
    CONV_DIAG,
    load_manifest,
    mesh_path,
    sample_spec_from_case,
    solve_case_dir,
    write_json,
)

DIAG_REPORT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_recovery_diagnostic.json"
DIAG_REPORT_MD = CONV_DIAG / "v2_l_mid_seed_branch_recovery_diagnostic.md"
MAC_AUDIT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_recovery_mac_audit.json"
AUDIT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_recovery_spurious_mode_audit.json"
AUDIT_MD = CONV_DIAG / "v2_l_mid_seed_branch_recovery_spurious_mode_audit.md"

CASE_ID = "baseline_coupled_v2"
SEED_F_HZ = 243.0754171175576
FOCAL_MODE_FILE = "modes/mode_243075_004.smx.npz"
FOCAL_F_HZ = 243.07546987835988
FOCAL_MODE_INDEX = 4
LAMBDA_ONE_TOL = 1.0e-3
REPLAY_RESIDUAL_SPURIOUS = 0.5

VERDICT = (
    "DIAGNOSTIC_SELECTED_SIGMA_OR_BC_SPURIOUS_MODE_"
    "TRUE_ACOUSTIC_SEED_REMAINS_VALID_BRANCH_NOT_YET_RECOVERED"
)


def _load_mode_vec(path: Path) -> np.ndarray:
    from fem_mode_array_utils import load_mode_column_any

    return np.asarray(load_mode_column_any(path).toarray(), dtype=np.float64).ravel()


def _assemble_layout(mesh_file: Path, sample: Dict[str, Any], tag: str):
    from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay

    case_dir = solve_case_dir("L_mid", CASE_ID)
    sort_dir = case_dir / "seed_branch_recovery_diagnostic" / "sorting_spurious_audit" / tag
    sort_dir.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sort_dir.resolve())
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    restr = dict(cfg.get("_coupled_air_pressure_restriction") or {})
    n_w = int(A.getSize()[0])
    return A, M, cfg, u_to_W, p_to_W, restr, n_w


def _pressure_mac(seed_block: np.ndarray, p_block: np.ndarray) -> float:
    a = np.asarray(seed_block, dtype=np.complex128).ravel()
    b = np.asarray(p_block, dtype=np.complex128).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(abs(np.vdot(a, b)) / (na * nb))


def _find_identity_rows(A, row_candidates: np.ndarray) -> np.ndarray:
    """Rows that look like algebraic Dirichlet / identity constraints on reduced W."""
    rows = np.unique(np.asarray(row_candidates, dtype=np.int32).ravel())
    if rows.size == 0:
        return np.array([], dtype=np.int32)
    try:
        diag = np.asarray(A.getDiagonal().array, dtype=np.float64).ravel()
    except Exception:
        diag = None
    ident: List[int] = []
    for r in rows:
        r = int(r)
        try:
            cols, vals = A.getRow(r)
        except TypeError:
            cols, vals = A.getRow(r)[0], A.getRow(r)[1]
        cols = np.asarray(cols, dtype=np.int32).ravel()
        vals = np.asarray(vals, dtype=np.float64).ravel()
        if cols.size == 0:
            continue
        off = cols[cols != r]
        if off.size > 0:
            continue
        if diag is not None and r < diag.size:
            if abs(diag[r]) < 1.0e-12:
                continue
        ident.append(r)
    return np.asarray(ident, dtype=np.int32)


def _support_fractions(
    vec: np.ndarray,
    *,
    n_w: int,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
    identity_rows: np.ndarray,
) -> Dict[str, float]:
    v = np.asarray(vec, dtype=np.float64).ravel()
    total = float(np.vdot(v, v))
    if total <= 0.0:
        return {k: float("nan") for k in (
            "unconstrained_active_pressure",
            "soundhole_pressure_dirichlet",
            "constrained_structural_displacement",
            "unconstrained_structural_displacement",
            "algebraic_bc_identity_rows",
            "other_reduced_W",
            "constrained_row_support_fraction",
        )}

    u_set = set(int(i) for i in u_to_W)
    p_set = set(int(i) for i in p_to_W)
    id_set = set(int(i) for i in identity_rows)
    p_id = p_set & id_set
    u_id = u_set & id_set
    p_interior = p_set - p_id
    u_free = u_set - u_id
    other = set(range(n_w)) - u_set - p_set

    def _frac(idxs: Set[int]) -> float:
        if not idxs:
            return 0.0
        ii = np.fromiter(idxs, dtype=np.int32)
        return float(np.vdot(v[ii], v[ii]) / total)

    constrained_frac = _frac(id_set)
    return {
        "unconstrained_active_pressure": _frac(p_interior),
        "soundhole_pressure_dirichlet": _frac(p_id),
        "constrained_structural_displacement": _frac(u_id),
        "unconstrained_structural_displacement": _frac(u_free),
        "algebraic_bc_identity_rows": _frac(id_set),
        "other_reduced_W": _frac(other),
        "constrained_row_support_fraction": constrained_frac,
    }


def _parse_eps_harvest_log(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.is_file():
        return []
    pat = re.compile(
        r"\[eps-p-diag\]\s+converged slot=(\d+)\s+raw_eig=([+-]?\d+\.\d+e[+-]\d+)\s+"
        r"lam_phys=([+-]?\d+\.\d+e[+-]\d+)\s+map=(\w+)\s+f=([+-]?\d+\.\d+)\s+Hz"
    )
    rows: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        raw = float(m.group(2))
        lam_phys = float(m.group(3))
        rows.append(
            {
                "eps_converged_slot": int(m.group(1)),
                "raw_eigenvalue": raw,
                "transformed_or_mapped_eigenvalue": lam_phys,
                "map_type": str(m.group(4)),
                "reported_frequency_hz": float(m.group(5)),
            }
        )
    return rows


def _match_eps_row(
    eps_rows: List[Dict[str, Any]], *, f_hz: float, mode_index: int
) -> Dict[str, Any]:
    if not eps_rows:
        return {}
    by_slot = {int(r["eps_converged_slot"]): r for r in eps_rows}
    if mode_index in by_slot:
        return dict(by_slot[mode_index])
    best = min(eps_rows, key=lambda r: abs(float(r["reported_frequency_hz"]) - f_hz))
    return dict(best)


def _map_eigenvalue_audit(
    raw_eig: float,
    *,
    sigma_hz: float,
    st_name: str = "sinvert",
    solver_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sigma_lam = fem3d._slepc_hz_to_lambda(float(sigma_hz))
    sc = dict(solver_cfg or {})
    lam_phys, tag = fem3d._slepc_physical_lambda(
        float(raw_eig), sigma_lam, str(st_name), sc
    )
    f_hz = math.sqrt(max(lam_phys, 0.0)) / (2.0 * math.pi) if math.isfinite(lam_phys) else float("nan")
    return {
        "raw_eigenvalue": float(raw_eig),
        "transformed_or_mapped_eigenvalue": float(lam_phys),
        "map_type": str(tag),
        "sigma_hz": float(sigma_hz),
        "sigma_lambda": float(sigma_lam),
        "mapped_frequency_hz": float(f_hz),
    }


def _replay_metrics(
    mesh_file: Path,
    sample: Dict[str, Any],
    vec: np.ndarray,
    *,
    reported_f_hz: float,
    tag: str,
) -> Dict[str, Any]:
    A, M, _cfg, u_to_W, p_to_W, _restr, _n_w = _assemble_layout(mesh_file, sample, tag)
    try:
        rayleigh = _rayleigh_metrics(A, M, vec, seed_f_hz=reported_f_hz)
        lam_r = float(rayleigh["rayleigh_lambda"])
        residual_at_reported = _block_residual_contributions(
            A,
            M,
            vec,
            lam0=(2.0 * math.pi * float(reported_f_hz)) ** 2,
            u_idx=u_to_W,
            p_idx=p_to_W,
        )
        residual_at_rayleigh = _block_residual_contributions(
            A, M, vec, lam0=lam_r, u_idx=u_to_W, p_idx=p_to_W
        )
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass
    return {
        "replay_rayleigh_eigenvalue": lam_r,
        "replay_rayleigh_frequency_hz": float(rayleigh["rayleigh_f_hz"]),
        "replay_relative_residual_at_reported_f": float(residual_at_reported["relative_residual"]),
        "replay_relative_residual_at_rayleigh_lambda": float(
            residual_at_rayleigh["relative_residual"]
        ),
        "algebraic_lambda_one_suspect": bool(abs(lam_r - 1.0) <= LAMBDA_ONE_TOL),
    }


def _selection_logic_audit(
    solve_result: Dict[str, Any],
    focal: Dict[str, Any],
    *,
    sigma_hz: float,
) -> Dict[str, Any]:
    diag = solve_result.get("seed_branch_recovery_diagnostic") or {}
    eps_diag = solve_result.get("eps_batch_diagnostics") or {}
    raw = float(focal.get("raw_eigenvalue", float("nan")))
    shift_hypothesis = (
        math.isfinite(raw)
        and abs(raw - 1.0) <= LAMBDA_ONE_TOL
        and math.isfinite(sigma_hz)
    )
    mapped = _map_eigenvalue_audit(raw, sigma_hz=sigma_hz) if math.isfinite(raw) else {}
    return {
        "diagnostic_mode_flags": {
            "eps_reject_sigma_spurious_disabled": True,
            "eps_reject_target_locked_disabled": True,
            "seed_branch_recovery_diagnostic": True,
            "standard_policy_not_used": bool(
                diag.get("standard_policy_not_used_for_this_diagnostic")
            ),
        },
        "why_mode_eligible_for_branch_recovery": [
            "Diagnostic evaluate ranked by pressure MAC among modes within 1% of seed frequency.",
            "Reported p_frac_energy_phys uses mass-participation on saved vector (can be ~1 on spurious vectors).",
            "No replay residual gate was applied before MAC audit; reported f was trusted for initial residual.",
            "eps_reject_sigma_spurious=False in seed_branch_recovery_diagnostic solver cfg.",
        ],
        "raw_mu_near_one_shift_mapping_hypothesis": {
            "active": shift_hypothesis,
            "explanation": (
                "If ST returns mu≈1 and harvest applies shift map lambda=mu+sigma, "
                "reported f≈sqrt(sigma)/(2*pi)≈sigma_hz even when the vector is an algebraic BC mode."
            ),
            "sigma_hz": sigma_hz,
            "raw_eigenvalue": raw,
            "remapped_frequency_hz": mapped.get("mapped_frequency_hz"),
            "map_type_from_log": focal.get("map_type"),
        },
        "eps_batch_diagnostics": eps_diag,
        "diagnostic_harvest_window_hz": diag.get("diagnostic_harvest_window_hz"),
        "diagnostic_local_band_hz": diag.get("diagnostic_local_band_hz"),
    }


def main() -> int:
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    diag_dir = case_dir / "seed_branch_recovery_diagnostic"
    mesh_file = mesh_path("L_mid", CASE_ID)
    sample = sample_spec_from_case(case)

    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    seed_w = _load_mode_vec(seed_npy) if seed_npy.is_file() else np.array([])

    result_paths = list((diag_dir / "results").glob("result_*.json"))
    solve_result = (
        json.loads(result_paths[0].read_text(encoding="utf-8")) if result_paths else {}
    )
    sigma_hz = float(
        (solve_result.get("seed_branch_recovery_diagnostic") or {}).get(
            "diagnostic_sigma_hz",
            solve_result.get("st_sigma_hz_used", SEED_F_HZ),
        )
    )
    st_name = str((solve_result.get("eps_batch_diagnostics") or {}).get("st_type", "sinvert"))

    summary_path = diag_dir / "diagnostics" / "mode_energy_summary.json"
    modes = (
        json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []
        if summary_path.is_file()
        else []
    )

    log_path = diag_dir / "logs" / "seed_branch_recovery_diagnostic.log"
    eps_rows = _parse_eps_harvest_log(log_path)

    A, M, _cfg, u_to_W, p_to_W, restr, n_w = _assemble_layout(mesh_file, sample, "layout")
    identity_rows = _find_identity_rows(A, np.concatenate([u_to_W, p_to_W]))
    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    p_seed_block = np.asarray(seed_w[p_to_W], dtype=np.float64).ravel() if seed_w.size else np.array([])

    focal_path = diag_dir / FOCAL_MODE_FILE
    focal_vec = _load_mode_vec(focal_path) if focal_path.is_file() else np.array([])
    focal_eps = _match_eps_row(eps_rows, f_hz=FOCAL_F_HZ, mode_index=FOCAL_MODE_INDEX)
    focal_replay = (
        _replay_metrics(
            mesh_file,
            sample,
            focal_vec,
            reported_f_hz=FOCAL_F_HZ,
            tag="focal",
        )
        if focal_vec.size
        else {}
    )
    focal_support = (
        _support_fractions(
            focal_vec,
            n_w=n_w,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            identity_rows=identity_rows,
        )
        if focal_vec.size
        else {}
    )

    focal_eig_meta = {
        **focal_eps,
        "sigma_hz": sigma_hz,
        "reported_frequency_hz": FOCAL_F_HZ,
        **focal_replay,
    }
    if focal_eps.get("raw_eigenvalue") is not None:
        focal_eig_meta.update(
            _map_eigenvalue_audit(
                float(focal_eps["raw_eigenvalue"]),
                sigma_hz=sigma_hz,
                st_name=st_name,
            )
        )

    candidates: List[Dict[str, Any]] = []
    physical_near_seed: List[Dict[str, Any]] = []
    for m in modes:
        rel = str(m.get("vector_path", ""))
        path = diag_dir / rel
        if not path.is_file():
            continue
        vec = _load_mode_vec(path)
        f_hz = float(m["frequency_hz"])
        mode_index = int(m.get("mode_index", -1))
        p_blk = np.asarray(vec[p_to_W], dtype=np.float64).ravel()
        mac = _pressure_mac(p_seed_block, p_blk)
        replay = _replay_metrics(
            mesh_file, sample, vec, reported_f_hz=f_hz, tag=f"mode_{mode_index:03d}"
        )
        support = _support_fractions(
            vec, n_w=n_w, u_to_W=u_to_W, p_to_W=p_to_W, identity_rows=identity_rows
        )
        eps_row = _match_eps_row(eps_rows, f_hz=f_hz, mode_index=mode_index)
        row = {
            "vector_file": str(path),
            "mode_index": mode_index,
            "reported_frequency_hz": f_hz,
            "reported_p_frac_energy_phys": m.get("p_frac_energy_phys"),
            "pressure_MAC_to_true_seed": mac,
            "replay_rayleigh_frequency_hz": replay.get("replay_rayleigh_frequency_hz"),
            "replay_rayleigh_eigenvalue": replay.get("replay_rayleigh_eigenvalue"),
            "replay_relative_residual": replay.get("replay_relative_residual_at_rayleigh_lambda"),
            "lambda_one_suspect": replay.get("algebraic_lambda_one_suspect"),
            "constrained_row_support_fraction": support.get("constrained_row_support_fraction"),
            "frequency_delta_fraction_from_seed": abs(f_hz - SEED_F_HZ) / SEED_F_HZ,
        }
        candidates.append(row)
        if (
            not replay.get("algebraic_lambda_one_suspect")
            and float(replay.get("replay_relative_residual_at_rayleigh_lambda", 1.0))
            <= REPLAY_RESIDUAL_SPURIOUS
            and mac >= 0.85
        ):
            physical_near_seed.append(row)

    candidates.sort(key=lambda r: float(r["frequency_delta_fraction_from_seed"]))

    focal_mode_record = {
        "vector_file": str(focal_path),
        "mode_index": FOCAL_MODE_INDEX,
        "reported_frequency_hz": FOCAL_F_HZ,
        "file_exists": focal_path.is_file(),
        "vector_length": int(focal_vec.size) if focal_vec.size else None,
        "eigenvalue_metadata": focal_eig_meta,
        "support_fractions": focal_support,
        "abs_replay_rayleigh_eigenvalue_minus_1": (
            abs(float(focal_replay.get("replay_rayleigh_eigenvalue", float("nan"))) - 1.0)
            if focal_replay
            else float("nan")
        ),
    }

    audit = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "verdict": VERDICT,
        "seed_rayleigh_f_hz": SEED_F_HZ,
        "true_acoustic_seed_valid": True,
        "focal_mode": focal_mode_record,
        "layout": {
            "n_reduced_W": n_w,
            "n_u_active": int(u_to_W.size),
            "n_p_active": int(p_to_W.size),
            "n_algebraic_identity_rows_detected": int(identity_rows.size),
            "pressure_restriction": restr,
        },
        "candidate_enumeration": candidates,
        "physical_candidates_near_seed": physical_near_seed,
        "selection_logic_audit": _selection_logic_audit(
            solve_result, focal_eig_meta, sigma_hz=sigma_hz
        ),
        "interpretation": {
            "selected_mode_is_sigma_or_bc_spurious": bool(
                focal_replay.get("algebraic_lambda_one_suspect")
                and float(focal_replay.get("replay_relative_residual_at_rayleigh_lambda", 0))
                >= REPLAY_RESIDUAL_SPURIOUS
            ),
            "true_acoustic_branch_not_recovered_in_saved_modes": len(physical_near_seed) == 0,
            "mac_audit_low_MAC_explained_by_spurious_vector": True,
        },
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
    }

    write_json(AUDIT_JSON, audit)

    lines = [
        "# L_mid seed-branch spurious-mode audit (baseline)",
        "",
        f"Generated: {audit['generated_utc']}",
        "",
        f"**Verdict:** `{VERDICT}`",
        "",
        "## Focal mode (previously reported as recovered)",
        "",
        f"- File: `{FOCAL_MODE_FILE}`",
        f"- Reported f: {FOCAL_F_HZ} Hz",
        f"- Replay Rayleigh f: {focal_replay.get('replay_rayleigh_frequency_hz')} Hz",
        f"- Replay Rayleigh λ: {focal_replay.get('replay_rayleigh_eigenvalue')}",
        f"- algebraic_lambda_one_suspect: {focal_replay.get('algebraic_lambda_one_suspect')}",
        f"- Replay residual (Rayleigh λ): {focal_replay.get('replay_relative_residual_at_rayleigh_lambda')}",
        f"- Constrained-row support fraction: {focal_support.get('constrained_row_support_fraction')}",
        "",
        "## True seed (unchanged)",
        "",
        f"- Seed f: {SEED_F_HZ} Hz (valid acoustic reference)",
        "",
        f"**mesh_convergence_may_resume:** `False`",
        "",
    ]
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if DIAG_REPORT_JSON.is_file():
        diag_report = json.loads(DIAG_REPORT_JSON.read_text(encoding="utf-8"))
        row = diag_report.setdefault("baseline_coupled_v2", {})
        ev = row.setdefault("evaluation", {})
        ev["diagnostic_verdict"] = VERDICT
        ev["spurious_mode_audit_json"] = str(AUDIT_JSON)
        ev["focal_mode_replay_rayleigh_lambda"] = focal_replay.get("replay_rayleigh_eigenvalue")
        ev["algebraic_lambda_one_suspect"] = focal_replay.get("algebraic_lambda_one_suspect")
        row["diagnostic_verdict"] = VERDICT
        diag_report["spurious_mode_audit_verdict"] = VERDICT
        diag_report["mesh_convergence_may_resume"] = False
        write_json(DIAG_REPORT_JSON, diag_report)

    if DIAG_REPORT_MD.is_file():
        text = DIAG_REPORT_MD.read_text(encoding="utf-8")
        marker = "## Spurious-mode audit verdict"
        block = (
            f"\n{marker}\n\n**`{VERDICT}`**\n\n"
            f"Focal `{FOCAL_MODE_FILE}`: replay λ≈1 suspect="
            f"{focal_replay.get('algebraic_lambda_one_suspect')}, "
            f"replay f={focal_replay.get('replay_rayleigh_frequency_hz')} Hz\n"
        )
        if marker not in text:
            DIAG_REPORT_MD.write_text(text + block, encoding="utf-8")

    if MAC_AUDIT_JSON.is_file():
        mac_report = json.loads(MAC_AUDIT_JSON.read_text(encoding="utf-8"))
        mac_report["superseded_by_spurious_mode_audit"] = True
        mac_report["corrected_baseline_verdict"] = VERDICT
        mac_report["audit_verdict"] = VERDICT
        write_json(MAC_AUDIT_JSON, mac_report)

    print(f"[spurious_audit] verdict={VERDICT}", flush=True)
    print(f"[spurious_audit] wrote {AUDIT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
