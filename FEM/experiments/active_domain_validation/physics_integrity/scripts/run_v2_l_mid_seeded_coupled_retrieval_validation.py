#!/usr/bin/env python3
"""
L_mid seeded coupled-v2 EPS retrieval validation (true acoustic reference).

One targeted EPS solve per case using validated acoustic_coupled_seed.npy.
No locator rerun, no remesh, no L_prod/L_check.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mode_diagnostics import (
    compute_mass_energy_participation,
    diagnose_mixed_mode,
    merge_scaling_metadata,
    pressure_subspace_mac,
)
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
from v2_sensitivity_common import REPO_ROOT, hz_result_tag

REPORT_JSON = CONV_DIAG / "v2_l_mid_seeded_coupled_retrieval_validation.json"
REPORT_MD = CONV_DIAG / "v2_l_mid_seeded_coupled_retrieval_validation.md"

ACOUSTIC_CASES = ("baseline_coupled_v2", "hole_radius_large")
ACOUSTIC_REFERENCE_SOURCE = "acoustic_only_locator_eigenvector"
HARVEST_HALF_WIDTH_HZ = 18.0
NUM_MODES = 32
FREQ_TOL_FRAC = 0.01
MAC_TOL = 0.85
REPLAY_RESIDUAL_OK = 0.05


def _seed_paths(case_dir: Path) -> Dict[str, Path]:
    return {
        "seed_npy": case_dir / "diagnostics" / "acoustic_coupled_seed.npy",
        "seed_meta": case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json",
    }


def _retrieval_case_dir(case_dir: Path) -> Path:
    return case_dir / "seeded_retrieval"


def _load_seed(case_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    paths = _seed_paths(case_dir)
    if not paths["seed_npy"].is_file():
        raise FileNotFoundError(paths["seed_npy"])
    seed = np.asarray(np.load(str(paths["seed_npy"])), dtype=np.float64).ravel()
    meta = (
        json.loads(paths["seed_meta"].read_text(encoding="utf-8"))
        if paths["seed_meta"].is_file()
        else {}
    )
    return seed, meta


def _run_seeded_solve(
    sample: Dict[str, Any],
    mesh_file: Path,
    *,
    target_hz: float,
    harvest_lo: float,
    harvest_hi: float,
    seed_npy: Path,
    retrieval_dir: Path,
) -> Tuple[int, Path, Dict[str, Any]]:
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    sample_json = retrieval_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    log_path = retrieval_dir / "logs" / "seeded_coupled_retrieval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mpiexec",
        "-n",
        "1",
        sys.executable,
        str(SCRIPT_DIR / "v2_sensitivity_solve.py"),
        "--sample-id",
        str(sample["id"]),
        "--mesh",
        str(mesh_file.resolve()),
        "--sample-json",
        str(sample_json.resolve()),
        "--case-dir",
        str(retrieval_dir.resolve()),
        "--target-hz",
        str(float(target_hz)),
        "--harvest-lo-hz",
        str(float(harvest_lo)),
        "--harvest-hi-hz",
        str(float(harvest_hi)),
        "--num-modes",
        str(int(NUM_MODES)),
        "--reference-f-hz",
        str(float(target_hz)),
        "--select-by-energy",
        "--eps-seed-npy",
        str(seed_npy.resolve()),
    ]
    with open(log_path, "w", encoding="utf-8") as logf:
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT).returncode
    result_path = retrieval_dir / "results" / f"result_{hz_result_tag(target_hz)}.json"
    result: Dict[str, Any] = {}
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    return int(rc), log_path, result


def _parse_eps_log(log_path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not log_path.is_file():
        return out
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "continuation_seed_source=" in text or "continuation_seed_applied" in text:
        out["log_shows_continuation_seed"] = True
    m = re.search(r"continuation_seed_source=(\S+)", text)
    if m:
        out["log_continuation_seed_source"] = m.group(1)
    m2 = re.search(r"seed_vector_length=(\d+)", text)
    if m2:
        out["log_seed_vector_length"] = int(m2.group(1))
    return out


def _assemble_operators(mesh_file: Path, sample: Dict[str, Any], sorting_tag: str):
    import fem_main_3d as fem3d

    from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay

    sort_dir = _retrieval_case_dir(solve_case_dir("L_mid", str(sample["id"]))) / "sorting_replay" / sorting_tag
    sort_dir.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sort_dir.resolve())
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    return A, M, u_to_W, p_to_W


def _load_mode_vec(retrieval_dir: Path, rel_path: str) -> np.ndarray:
    from fem_mode_array_utils import load_mode_column_any

    col = load_mode_column_any(retrieval_dir / rel_path)
    return np.asarray(col.toarray(), dtype=np.float64).ravel()


def _audit_energy_classification(
    retrieval_dir: Path,
    mesh_file: Path,
    sample: Dict[str, Any],
    recovered: Dict[str, Any],
    seed: np.ndarray,
    *,
    u_to_W: np.ndarray,
    p_to_W: np.ndarray,
) -> Dict[str, Any]:
    vec = _load_mode_vec(retrieval_dir, str(recovered["vector_path"]))
    gnhep = merge_scaling_metadata(retrieval_dir)
    A, M, _, _ = _assemble_operators(
        mesh_file, sample, f"energy_audit_{int(recovered['mode_index'])}"
    )
    try:
        energy = compute_mass_energy_participation(
            vec, M, A, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep
        )
        diag = diagnose_mixed_mode(
            vec,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
            gnhep=gnhep,
            frequency_hz=float(recovered["frequency_hz"]),
        )
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    p_seed = np.asarray(seed[p_idx], dtype=np.float64)
    p_rec = np.asarray(vec[p_idx], dtype=np.float64)
    ns = float(np.linalg.norm(p_seed))
    nr = float(np.linalg.norm(p_rec))
    p_frac_reported = float(recovered.get("p_frac_energy_phys", float("nan")))
    p_frac_replay = float(energy.get("p_frac_energy_phys", float("nan")))
    class_reported = str(recovered.get("mode_class_physical_energy", ""))
    class_diag = str(diag.get("mode_class", ""))

    suspect = False
    reasons: List[str] = []
    if float(recovered.get("pressure_MAC_to_true_acoustic_reference", 0.0)) >= MAC_TOL:
        if p_frac_reported < 0.05 and p_frac_replay < 0.05:
            suspect = True
            reasons.append("high MAC but both reported and replay p_frac near zero")
        if class_reported == "structural_dominated" and p_frac_replay >= 0.15:
            suspect = True
            reasons.append("structural_dominated label despite replay p_frac >= 0.15")
        if math.isfinite(p_frac_reported) and math.isfinite(p_frac_replay):
            if abs(p_frac_reported - p_frac_replay) > 0.25:
                suspect = True
                reasons.append("reported vs replay p_frac_energy_phys disagree")

    return {
        "p_frac_energy_phys_reported": p_frac_reported,
        "p_frac_energy_phys_replay": p_frac_replay,
        "E_air_phys_replay": energy.get("E_air_phys"),
        "E_struct_phys_replay": energy.get("E_struct_phys"),
        "mode_class_physical_energy_reported": class_reported,
        "mode_class_diagnose_mixed_mode": class_diag,
        "p_norm_true_seed_on_active_p": ns,
        "p_norm_recovered_on_active_p": nr,
        "p_amplitude_ratio_recovered_over_seed": (nr / ns) if ns > 0 else float("nan"),
        "energy_classification_suspect": suspect,
        "energy_classification_suspect_reasons": reasons,
        "mode_class_trustworthy_for_branch_tracking": not suspect,
    }


def _evaluate_recovery(
    retrieval_dir: Path,
    mesh_file: Path,
    sample: Dict[str, Any],
    seed: np.ndarray,
    seed_meta: Dict[str, Any],
    solve_result: Dict[str, Any],
    log_path: Path,
) -> Dict[str, Any]:
    seed_rayleigh_f_hz = float(
        seed_meta.get("locator_frequency_hz", solve_result.get("target_hz", float("nan")))
    )
    summary_path = retrieval_dir / "diagnostics" / "mode_energy_summary.json"
    modes = []
    if summary_path.is_file():
        modes = json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []

    u_to_W = np.asarray(solve_result.get("u_to_W") or [], dtype=np.int32).ravel()
    p_to_W = np.asarray(solve_result.get("p_to_W") or [], dtype=np.int32).ravel()
    if u_to_W.size == 0 or p_to_W.size == 0:
        A, M, u_to_W, p_to_W = _assemble_operators(mesh_file, sample, "map_fallback")
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass

    ranked: List[Dict[str, Any]] = []
    for m in modes:
        try:
            vec = _load_mode_vec(retrieval_dir, str(m["vector_path"]))
        except Exception:
            continue
        f_hz = float(m["frequency_hz"])
        mac = pressure_subspace_mac(seed, vec, p_to_W)
        d_f_rel = (
            abs(f_hz - seed_rayleigh_f_hz) / seed_rayleigh_f_hz
            if math.isfinite(seed_rayleigh_f_hz) and seed_rayleigh_f_hz > 0
            else float("inf")
        )
        ranked.append(
            {
                **m,
                "pressure_MAC_to_true_acoustic_reference": float(mac),
                "frequency_delta_from_seed_rayleigh_hz": f_hz - seed_rayleigh_f_hz,
                "frequency_delta_fraction_from_seed_rayleigh": float(d_f_rel),
            }
        )

    ranked.sort(key=lambda r: -float(r["pressure_MAC_to_true_acoustic_reference"]))
    in_freq = [
        r
        for r in ranked
        if float(r["frequency_delta_fraction_from_seed_rayleigh"]) <= FREQ_TOL_FRAC
    ]
    pool = in_freq if in_freq else ranked
    recovered = pool[0] if pool else {}

    replay: Dict[str, Any] = {}
    if recovered and recovered.get("vector_path"):
        vec = _load_mode_vec(retrieval_dir, str(recovered["vector_path"]))
        f_hz = float(recovered["frequency_hz"])
        lam0 = (2.0 * math.pi * f_hz) ** 2
        A, M, u_idx, p_idx = _assemble_operators(
            mesh_file, sample, f"replay_{hz_result_tag(f_hz)}"
        )
        try:
            residual = _block_residual_contributions(
                A, M, vec, lam0=lam0, u_idx=u_idx, p_idx=p_idx
            )
            rayleigh = _rayleigh_metrics(A, M, vec, seed_f_hz=f_hz)
        finally:
            try:
                A.destroy()
                M.destroy()
            except Exception:
                pass
        replay = {
            "replay_relative_residual_of_recovered_mode": float(residual["relative_residual"]),
            "replay_rayleigh_f_hz": float(rayleigh["rayleigh_f_hz"]),
            "replay_delta_rayleigh_hz": float(rayleigh["delta_rayleigh_from_seed_hz"]),
        }

    mac = float(recovered.get("pressure_MAC_to_true_acoustic_reference", float("nan")))
    d_frac = float(recovered.get("frequency_delta_fraction_from_seed_rayleigh", float("inf")))
    rel_res = float(replay.get("replay_relative_residual_of_recovered_mode", float("nan")))
    recovery_ok = (
        recovered
        and math.isfinite(mac)
        and mac >= MAC_TOL
        and math.isfinite(d_frac)
        and d_frac <= FREQ_TOL_FRAC
        and math.isfinite(rel_res)
        and rel_res <= REPLAY_RESIDUAL_OK
    )

    energy_audit: Dict[str, Any] = {}
    if recovered:
        energy_audit = _audit_energy_classification(
            retrieval_dir,
            mesh_file,
            sample,
            recovered,
            seed,
            u_to_W=u_to_W,
            p_to_W=p_to_W,
        )

    eps_seed = solve_result.get("eps_seed") or {}
    eps_log = _parse_eps_log(log_path)
    eps_diag = solve_result.get("eps_batch_diagnostics") or {}
    eps_initial_space_set = bool(
        eps_seed.get("eps_initial_space_set")
        or eps_diag.get("continuation_seed_applied")
        or eps_log.get("log_shows_continuation_seed")
    )

    if not eps_initial_space_set:
        verdict = "EPS_INITIAL_SPACE_NOT_VERIFIED"
    elif not recovery_ok:
        verdict = "SEEDED_COUPLED_BRANCH_NOT_RECOVERED"
    elif energy_audit.get("energy_classification_suspect"):
        verdict = "SEEDED_BRANCH_RECOVERED_BUT_ENERGY_CLASSIFICATION_SUSPECT"
    else:
        verdict = "SEEDED_COUPLED_BRANCH_RECOVERED"

    return {
        "seed_rayleigh_f_hz": seed_rayleigh_f_hz,
        "true_reference_source": ACOUSTIC_REFERENCE_SOURCE,
        "eps_solve": {
            "seed_file_used": eps_seed.get("seed_file_used"),
            "seed_vector_length": eps_seed.get("seed_vector_length"),
            "seed_layout_valid": eps_seed.get("seed_layout_valid"),
            "eps_initial_space_set": eps_initial_space_set,
            "eps_initial_space_norm": eps_seed.get("eps_initial_space_norm"),
            "target_hz": solve_result.get("target_hz"),
            "harvest_band_hz": solve_result.get("harvest_band_hz"),
            "nconv": solve_result.get("nconv"),
            "continuation_seed_applied": eps_diag.get("continuation_seed_applied"),
            "log_parse": eps_log,
        },
        "recovered_mode": {
            "recovered_mode_frequency_hz": recovered.get("frequency_hz"),
            "frequency_delta_from_seed_rayleigh_hz": recovered.get(
                "frequency_delta_from_seed_rayleigh_hz"
            ),
            "pressure_MAC_to_true_acoustic_reference": mac,
            "p_frac_energy_phys": recovered.get("p_frac_energy_phys"),
            "mode_class_physical_energy": recovered.get("mode_class_physical_energy"),
            "mode_index": recovered.get("mode_index"),
            "vector_path": recovered.get("vector_path"),
            **replay,
        },
        "recovery_success_by_true_reference_metrics": recovery_ok,
        "energy_classification_audit": energy_audit,
        "top_modes_by_pressure_MAC": ranked[:8],
        "prior_mixed_mode_audit_note": (
            "Prior L_MID_ACOUSTIC_BRANCH_CONTINUES_AS_SINGLE_MIXED_MODE verdict from "
            "proxy/circular saved-mode analysis is superseded; true seed replay validates "
            "acoustic branch under physical coupling. Remaining issue is EPS harvest / "
            "energy classification on saved modes."
        ),
        "diagnostic_verdict": verdict,
    }


def _process_case(manifest: Dict[str, Any], case_id: str) -> Dict[str, Any]:
    case = next(c for c in manifest["cases"] if str(c["id"]) == case_id)
    mesh_file = mesh_path("L_mid", case_id)
    case_dir = solve_case_dir("L_mid", case_id)
    retrieval_dir = _retrieval_case_dir(case_dir)
    sample = sample_spec_from_case(case)

    row: Dict[str, Any] = {"sample_id": case_id, "mesh_file": str(mesh_file)}
    if not mesh_file.is_file():
        row["status"] = "failed"
        row["error"] = f"missing mesh {mesh_file}"
        row["diagnostic_verdict"] = "SEEDED_COUPLED_BRANCH_NOT_RECOVERED"
        return row

    try:
        seed, seed_meta = _load_seed(case_dir)
    except FileNotFoundError as exc:
        row["status"] = "failed"
        row["error"] = str(exc)
        row["diagnostic_verdict"] = "EPS_INITIAL_SPACE_NOT_VERIFIED"
        return row

    if seed_meta.get("locator_pressure_reference_source") != ACOUSTIC_REFERENCE_SOURCE:
        row["status"] = "failed"
        row["error"] = "seed meta not tagged as acoustic_only_locator_eigenvector"
        row["diagnostic_verdict"] = "EPS_INITIAL_SPACE_NOT_VERIFIED"
        return row

    target_hz = float(seed_meta.get("locator_frequency_hz", float("nan")))
    harvest_lo = target_hz - HARVEST_HALF_WIDTH_HZ
    harvest_hi = target_hz + HARVEST_HALF_WIDTH_HZ

    rc, log_path, solve_result = _run_seeded_solve(
        sample,
        mesh_file,
        target_hz=target_hz,
        harvest_lo=harvest_lo,
        harvest_hi=harvest_hi,
        seed_npy=_seed_paths(case_dir)["seed_npy"],
        retrieval_dir=retrieval_dir,
    )
    row["solve_exit_code"] = rc
    row["retrieval_output_dir"] = str(retrieval_dir)
    row["evaluation"] = _evaluate_recovery(
        retrieval_dir, mesh_file, sample, seed, seed_meta, solve_result, log_path
    )
    row["diagnostic_verdict"] = row["evaluation"]["diagnostic_verdict"]
    row["status"] = "ok" if rc == 0 else f"solve_exit_{rc}"
    return row


def _write_md(report: Dict[str, Any]) -> None:
    lines = [
        "# L_mid seeded coupled retrieval validation",
        "",
        f"Generated: {report.get('generated_utc')}",
        "",
        "Targeted coupled-v2 EPS with true acoustic coupled-W seed. "
        "Branch recovery judged by pressure MAC and frequency vs true reference, "
        "not by legacy p_frac >= 0.85 alone.",
        "",
        f"**mesh_convergence_may_resume:** `{report.get('mesh_convergence_may_resume')}`",
        "",
    ]
    for cid in ACOUSTIC_CASES:
        row = (report.get("cases") or {}).get(cid) or {}
        ev = row.get("evaluation") or {}
        eps = ev.get("eps_solve") or {}
        rec = ev.get("recovered_mode") or {}
        en = ev.get("energy_classification_audit") or {}
        lines.append(f"## {cid}")
        lines.append("")
        lines.append(f"- **Verdict:** `{row.get('diagnostic_verdict')}`")
        lines.append(
            f"- EPS seed: file={eps.get('seed_file_used')} "
            f"set={eps.get('eps_initial_space_set')} norm={eps.get('eps_initial_space_norm')} "
            f"nconv={eps.get('nconv')} band={eps.get('harvest_band_hz')}"
        )
        lines.append(
            f"- Recovered: f={rec.get('recovered_mode_frequency_hz')} Hz "
            f"MAC={rec.get('pressure_MAC_to_true_acoustic_reference')} "
            f"p_frac={rec.get('p_frac_energy_phys')} class={rec.get('mode_class_physical_energy')}"
        )
        lines.append(
            f"- Replay residual={rec.get('replay_relative_residual_of_recovered_mode')} "
            f"Δf_seed={rec.get('frequency_delta_from_seed_rayleigh_hz')}"
        )
        if en.get("energy_classification_suspect"):
            lines.append(f"- **Energy classification suspect:** {en.get('energy_classification_suspect_reasons')}")
        lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    manifest = load_manifest()
    CONV_DIAG.mkdir(parents=True, exist_ok=True)

    cases: Dict[str, Any] = {}
    for cid in ACOUSTIC_CASES:
        print(f"[seeded_retrieval] {cid}", flush=True)
        cases[cid] = _process_case(manifest, cid)

    ok_verdicts = {
        "SEEDED_COUPLED_BRANCH_RECOVERED",
        "SEEDED_BRANCH_RECOVERED_BUT_ENERGY_CLASSIFICATION_SUSPECT",
    }
    both_recovered = all(
        str((cases.get(c) or {}).get("diagnostic_verdict")) in ok_verdicts for c in ACOUSTIC_CASES
    )
    both_mac = all(
        float(
            ((cases.get(c) or {}).get("evaluation") or {})
            .get("recovered_mode", {})
            .get("pressure_MAC_to_true_acoustic_reference", 0.0)
        )
        >= MAC_TOL
        for c in ACOUSTIC_CASES
    )

    report = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interpretation": (
            "True acoustic seed remains valid under physical coupling on L_mid. "
            "This stage tests whether EPS can retrieve the branch when seeded; "
            "p_frac-based labels on prior saves may be unreliable."
        ),
        "cases": cases,
        "recovery_criteria": {
            "frequency_tolerance_fraction": FREQ_TOL_FRAC,
            "pressure_MAC_minimum": MAC_TOL,
            "replay_relative_residual_maximum": REPLAY_RESIDUAL_OK,
        },
        "mesh_convergence_may_resume": bool(both_recovered and both_mac),
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
        "supersedes_proxy_verdict": (
            "L_MID_ACOUSTIC_BRANCH_CONTINUES_AS_SINGLE_MIXED_MODE from proxy audit is not "
            "the operative diagnosis; prefer "
            "L_MID_ACOUSTIC_SEED_REMAINS_VALID_UNDER_PHYSICAL_COUPLING_BUT_EPS_HARVEST_OR_ENERGY_CLASSIFICATION_FAILED "
            "when seeded retrieval fails."
        ),
    }
    write_json(REPORT_JSON, report)
    _write_md(report)
    print(f"[seeded_retrieval] wrote {REPORT_JSON}", flush=True)
    return 0 if both_recovered and both_mac else 1


if __name__ == "__main__":
    raise SystemExit(main())
