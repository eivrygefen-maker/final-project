#!/usr/bin/env python3
"""
Report-only MAC / representation audit for baseline seed-branch recovery diagnostic.

Reuses completed artifacts under seed_branch_recovery_diagnostic/ — no new eigensolves.
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
for _p in (SCRIPT_DIR, FEM_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mode_diagnostics import pressure_subspace_mac
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
AUDIT_JSON = CONV_DIAG / "v2_l_mid_seed_branch_recovery_mac_audit.json"
AUDIT_MD = CONV_DIAG / "v2_l_mid_seed_branch_recovery_mac_audit.md"

CASE_ID = "baseline_coupled_v2"
TARGET_F_HZ = 243.0754171175576
RECOVERED_F_HZ_REPORTED = 243.07546987835988
FREQ_TOL_FRAC = 0.01
MAC_TOL = 0.85
REPLAY_RESIDUAL_OK = 0.05


def _crc32(arr: np.ndarray) -> int:
    return int(zlib.crc32(np.asarray(arr, dtype=np.int32).tobytes()) & 0xFFFFFFFF)


def _map_stats(name: str, idx: np.ndarray, n_w: int) -> Dict[str, Any]:
    idx = np.asarray(idx, dtype=np.int32).ravel()
    if idx.size == 0:
        return {"name": name, "length": 0, "bounds_valid": False, "crc32": _crc32(idx)}
    lo, hi = int(idx.min()), int(idx.max())
    return {
        "name": name,
        "length": int(idx.size),
        "min": lo,
        "max": hi,
        "bounds_valid": bool(lo >= 0 and hi < int(n_w)),
        "crc32": _crc32(idx),
    }


def _pressure_block(vec: np.ndarray, p_to_W: np.ndarray) -> np.ndarray:
    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    return np.asarray(vec[p_idx], dtype=np.float64).ravel()


def _euclidean_mac(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(abs(np.vdot(a, b)) / (na * nb))


def _vector_stats(label: str, v: np.ndarray) -> Dict[str, Any]:
    v = np.asarray(v, dtype=np.float64).ravel()
    return {
        "label": label,
        "length": int(v.size),
        "l2_norm": float(np.linalg.norm(v)),
        "max_abs": float(np.max(np.abs(v))) if v.size else float("nan"),
    }


def _assemble_layout(mesh_file: Path, sample: Dict[str, Any], tag: str):
    import fem_main_3d as fem3d
    from v2_build_coupled_acoustic_seed import _assemble_reduced_coupled_replay

    case_dir = solve_case_dir("L_mid", CASE_ID)
    sort_dir = case_dir / "seed_branch_recovery_diagnostic" / "sorting_mac_audit" / tag
    sort_dir.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sort_dir.resolve())
    A, M, cfg = _assemble_reduced_coupled_replay(mesh_file, sample, coupling_enabled=True)
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    p_air = np.asarray(cfg["_coupled_air_p_air_collapsed_indices"], dtype=np.int32).ravel()
    restr = dict(cfg.get("_coupled_air_pressure_restriction") or {})
    n_w = int(A.getSize()[0])
    return A, M, cfg, u_to_W, p_to_W, p_air, restr, n_w


def _load_mode_vec(path: Path) -> np.ndarray:
    from fem_mode_array_utils import load_mode_column_any

    col = load_mode_column_any(path)
    arr = np.asarray(col.toarray(), dtype=np.float64).ravel()
    return arr


def _find_recovered_mode_entry(
    modes: List[Dict[str, Any]], diag_dir: Path, *, target_f: float
) -> Dict[str, Any]:
    best_row: Optional[Dict[str, Any]] = None
    best_df = float("inf")
    for m in modes:
        f_hz = float(m.get("frequency_hz", float("nan")))
        if not math.isfinite(f_hz):
            continue
        df = abs(f_hz - target_f)
        if df < best_df:
            best_df = df
            best_row = m
    if best_row is None:
        return {"file_exists": False}
    rel = str(best_row.get("vector_path", ""))
    path = diag_dir / rel
    out = {
        "recovered_mode_file": str(path),
        "recovered_mode_index": best_row.get("mode_index"),
        "recovered_mode_frequency_hz": float(best_row.get("frequency_hz")),
        "file_exists": path.is_file(),
        "vector_path_in_summary": rel,
        "frequency_delta_from_reported_hz": float(best_row["frequency_hz"]) - float(target_f),
        "summary_row": best_row,
    }
    if path.is_file():
        vec = _load_mode_vec(path)
        out["vector_length"] = int(vec.size)
    return out


def _replay_mode(
    mesh_file: Path,
    sample: Dict[str, Any],
    vec: np.ndarray,
    f_hz: float,
    tag: str,
) -> Dict[str, Any]:
    A, M, _cfg, u_to_W, p_to_W, _p_air, _restr, _n_w = _assemble_layout(
        mesh_file, sample, tag
    )
    lam0 = (2.0 * math.pi * float(f_hz)) ** 2
    try:
        residual = _block_residual_contributions(
            A, M, vec, lam0=lam0, u_idx=u_to_W, p_idx=p_to_W
        )
        rayleigh = _rayleigh_metrics(A, M, vec, seed_f_hz=f_hz)
    finally:
        try:
            A.destroy()
            M.destroy()
        except Exception:
            pass
    return {
        "relative_residual": float(residual["relative_residual"]),
        "rayleigh_frequency_hz": float(rayleigh["rayleigh_f_hz"]),
        "delta_rayleigh_from_f_hz": float(rayleigh["delta_rayleigh_from_seed_hz"]),
    }


def _assign_verdict(
    *,
    continuation: bool,
    mac_seed_recovered: float,
    d_frac: float,
    rel_res: float,
    mac_locator_seed: float,
    representation_suspect: bool,
) -> str:
    if not continuation:
        return "DIAGNOSTIC_SOLVER_NOT_APPLIED"
    if representation_suspect:
        return "DIAGNOSTIC_OUTPUT_MODE_REPRESENTATION_OR_SELECTION_SUSPECT"
    recovery_ok = (
        math.isfinite(mac_seed_recovered)
        and mac_seed_recovered >= MAC_TOL
        and math.isfinite(d_frac)
        and d_frac <= FREQ_TOL_FRAC
        and math.isfinite(rel_res)
        and rel_res <= REPLAY_RESIDUAL_OK
        and math.isfinite(mac_locator_seed)
        and mac_locator_seed >= 0.99
    )
    if recovery_ok:
        return "SEED_BRANCH_RECOVERED_IN_DIAGNOSTIC_MODE"
    if (
        math.isfinite(d_frac)
        and d_frac <= FREQ_TOL_FRAC
        and math.isfinite(mac_seed_recovered)
        and mac_seed_recovered < MAC_TOL
    ):
        return "DIAGNOSTIC_BRANCH_FREQUENCY_AND_ACOUSTIC_CLASS_RECOVERED_BUT_MAC_OR_REPLAY_MAPPING_INCONSISTENT"
    return "SEED_BRANCH_NOT_RECOVERED_EVEN_IN_DIAGNOSTIC_MODE"


def main() -> int:
    manifest = load_manifest()
    case = next(c for c in manifest["cases"] if str(c["id"]) == CASE_ID)
    case_dir = solve_case_dir("L_mid", CASE_ID)
    diag_dir = case_dir / "seed_branch_recovery_diagnostic"
    mesh_file = mesh_path("L_mid", CASE_ID)
    sample = sample_spec_from_case(case)

    seed_npy = case_dir / "diagnostics" / "acoustic_coupled_seed.npy"
    locator_npy = case_dir / "diagnostics" / "l_mid_true_ref" / "acoustic_locator_pressure.npy"
    seed_meta_path = case_dir / "diagnostics" / "acoustic_coupled_seed_meta.json"
    seed_meta = json.loads(seed_meta_path.read_text(encoding="utf-8")) if seed_meta_path.is_file() else {}

    result_paths = list((diag_dir / "results").glob("result_*.json"))
    solve_result = (
        json.loads(result_paths[0].read_text(encoding="utf-8")) if result_paths else {}
    )
    prior_eval = {}
    if DIAG_REPORT_JSON.is_file():
        prior_eval = json.loads(DIAG_REPORT_JSON.read_text(encoding="utf-8")).get(
            "baseline_coupled_v2", {}
        ).get("evaluation", {})

    summary_path = diag_dir / "diagnostics" / "mode_energy_summary.json"
    modes = (
        json.loads(summary_path.read_text(encoding="utf-8")).get("modes") or []
        if summary_path.is_file()
        else []
    )

    A_layout, M_layout, _cfg, u_to_W, p_to_W, p_air_collapsed, restr, n_w = _assemble_layout(
        mesh_file, sample, "layout"
    )

    seed_w = _load_mode_vec(seed_npy) if seed_npy.is_file() else np.array([])
    p_full = _load_mode_vec(locator_npy) if locator_npy.is_file() else np.array([])

    layout_validation = {
        "full_locator_pressure_length": int(p_full.size),
        "active_air_pressure_index_length": int(p_air_collapsed.size),
        "n_p_active": int(restr.get("n_p_active", p_to_W.size)),
        "n_u_active": int(restr.get("n_u_active", 0)),
        "n_reduced_W": int(n_w),
        "len_p_to_W": int(p_to_W.size),
        "p_to_W": _map_stats("p_to_W", p_to_W, n_w),
        "active_pressure_selection": _map_stats(
            "p_air_collapsed_indices", p_air_collapsed, int(p_full.size) if p_full.size else n_w
        ),
        "seed_vector_length": int(seed_w.size),
    }

    p_ref_from_locator = np.array([], dtype=np.float64)
    if p_full.size > 0 and p_air_collapsed.size > 0:
        hi = int(p_air_collapsed.max())
        if hi < p_full.size:
            p_ref_from_locator = np.asarray(p_full[p_air_collapsed], dtype=np.float64).ravel()
    p_ref_from_seed_block = _pressure_block(seed_w, p_to_W)

    recovered_entry = _find_recovered_mode_entry(
        modes, diag_dir, target_f=RECOVERED_F_HZ_REPORTED
    )
    recovered_w = (
        _load_mode_vec(Path(recovered_entry["recovered_mode_file"]))
        if recovered_entry.get("file_exists")
        else np.array([])
    )
    layout_validation["recovered_vector_length"] = int(recovered_w.size)

    p_recovered_block = _pressure_block(recovered_w, p_to_W) if recovered_w.size else np.array([])

    pressure_vectors = {
        "p_ref_from_archived_full_locator_restricted_to_active_air": _vector_stats(
            "locator_restricted", p_ref_from_locator
        ),
        "p_ref_from_coupled_W_seed_at_p_to_W": _vector_stats("seed_p_block", p_ref_from_seed_block),
        "p_recovered_from_saved_coupled_mode_at_p_to_W": _vector_stats(
            "recovered_p_block", p_recovered_block
        ),
    }

    mac_table = {
        "MAC_locator_restricted_seed_p_block": _euclidean_mac(p_ref_from_locator, p_ref_from_seed_block),
        "MAC_locator_restricted_recovered_p_block": _euclidean_mac(
            p_ref_from_locator, p_recovered_block
        ),
        "MAC_seed_p_block_recovered_p_block": _euclidean_mac(
            p_ref_from_seed_block, p_recovered_block
        ),
    }

    # Independent path via pressure_subspace_mac on full W vectors
    mac_via_p_to_W = {
        "MAC_seed_W_recovered_W": pressure_subspace_mac(seed_w, recovered_w, p_to_W),
        "MAC_seed_W_locator_restricted_embedded": float("nan"),
    }
    if p_ref_from_locator.size == p_to_W.size:
        locator_embed = np.zeros(n_w, dtype=np.float64)
        locator_embed[p_to_W] = p_ref_from_locator
        nrm = float(np.linalg.norm(locator_embed))
        if nrm > 0:
            locator_embed /= nrm
        mac_via_p_to_W["MAC_seed_W_locator_restricted_embedded"] = pressure_subspace_mac(
            seed_w, locator_embed, p_to_W
        )

    candidate_rows: List[Dict[str, Any]] = []
    for m in modes:
        rel = str(m.get("vector_path", ""))
        path = diag_dir / rel
        if not path.is_file():
            continue
        vec = _load_mode_vec(path)
        p_blk = _pressure_block(vec, p_to_W)
        f_hz = float(m["frequency_hz"])
        candidate_rows.append(
            {
                "frequency_hz": f_hz,
                "vector_file": str(path),
                "mode_index": m.get("mode_index"),
                "pressure_MAC_to_seed_p_block": _euclidean_mac(p_ref_from_seed_block, p_blk),
                "pressure_MAC_via_subspace_on_W": pressure_subspace_mac(seed_w, vec, p_to_W),
                "p_frac_energy_phys": m.get("p_frac_energy_phys"),
                "mode_class_physical_energy": m.get("mode_class_physical_energy"),
                "frequency_delta_fraction_from_seed": (
                    abs(f_hz - TARGET_F_HZ) / TARGET_F_HZ if TARGET_F_HZ > 0 else float("inf")
                ),
            }
        )
    candidate_rows.sort(
        key=lambda r: (
            float(r["frequency_delta_fraction_from_seed"]),
            -float(r["pressure_MAC_to_seed_p_block"]),
        )
    )

    reported_f = float(
        (prior_eval.get("recovered_mode") or {}).get("frequency_hz", RECOVERED_F_HZ_REPORTED)
    )
    reported_mac = float(
        prior_eval.get("recovered_mode", {}).get(
            "pressure_MAC_to_true_acoustic_reference", float("nan")
        )
    )
    near_reported = [
        r
        for r in candidate_rows
        if abs(float(r["frequency_hz"]) - reported_f) / reported_f <= 1.0e-4
    ]
    selection_audit = {
        "prior_reported_frequency_hz": reported_f,
        "prior_reported_MAC": reported_mac,
        "candidates_matching_reported_frequency": near_reported,
        "best_MAC_among_all_candidates": max(
            candidate_rows, key=lambda r: float(r["pressure_MAC_to_seed_p_block"])
        )
        if candidate_rows
        else None,
        "best_MAC_among_freq_within_1pct": max(
            (
                r
                for r in candidate_rows
                if float(r["frequency_delta_fraction_from_seed"]) <= FREQ_TOL_FRAC
            ),
            key=lambda r: float(r["pressure_MAC_to_seed_p_block"]),
            default={"pressure_MAC_to_seed_p_block": float("nan")},
        ),
    }

    continuation = bool(
        solve_result.get("continuation_seed_applied")
        or (solve_result.get("eps_seed") or {}).get("eps_initial_space_set")
    )

    mass_weighted_mac: Dict[str, float] = {}
    try:
        from petsc4py import PETSc

        def _mass_inner_on_pressure(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
            va = PETSc.Vec().create(comm=PETSc.COMM_SELF)
            va.setSizes(int(n_w))
            va.setUp()
            arr = np.zeros(int(n_w), dtype=np.float64)
            arr[p_to_W] = np.asarray(vec_a, dtype=np.float64).ravel()
            va.array[:] = arr
            vb = va.duplicate()
            arr_b = np.zeros(int(n_w), dtype=np.float64)
            arr_b[p_to_W] = np.asarray(vec_b, dtype=np.float64).ravel()
            vb.array[:] = arr_b
            Ma = M_layout.createVecRight()
            Mb = M_layout.createVecRight()
            M_layout.mult(va, Ma)
            M_layout.mult(vb, Mb)
            pa = np.asarray([Ma.array[i] for i in p_to_W], dtype=np.float64)
            pb = np.asarray([Mb.array[i] for i in p_to_W], dtype=np.float64)
            va.destroy()
            vb.destroy()
            Ma.destroy()
            Mb.destroy()
            na = float(np.sqrt(np.real(np.vdot(pa, pa))))
            nb = float(np.sqrt(np.real(np.vdot(pb, pb))))
            if na <= 0 or nb <= 0:
                return float("nan")
            return float(abs(np.vdot(pa, pb)) / (na * nb))

        mass_weighted_mac = {
            "MAC_mass_weighted_locator_restricted_seed_p_block": _mass_inner_on_pressure(
                p_ref_from_locator, p_ref_from_seed_block
            ),
            "MAC_mass_weighted_seed_p_block_recovered_p_block": _mass_inner_on_pressure(
                p_ref_from_seed_block, p_recovered_block
            ),
        }
    except Exception as exc:
        mass_weighted_mac = {"error": str(exc)}
    finally:
        try:
            A_layout.destroy()
            M_layout.destroy()
        except Exception:
            pass

    seed_replay = _replay_mode(mesh_file, sample, seed_w, TARGET_F_HZ, "replay_seed")
    recovered_replay = {}
    if recovered_w.size:
        recovered_replay = _replay_mode(
            mesh_file,
            sample,
            recovered_w,
            float(recovered_entry.get("recovered_mode_frequency_hz", reported_f)),
            "replay_recovered",
        )

    mac_seed_recovered = float(mac_table["MAC_seed_p_block_recovered_p_block"])
    d_frac = (
        abs(float(recovered_entry.get("recovered_mode_frequency_hz", float("nan"))) - TARGET_F_HZ)
        / TARGET_F_HZ
        if TARGET_F_HZ > 0
        else float("inf")
    )
    rel_res = float(recovered_replay.get("relative_residual", float("nan")))

    locator_seed_mac = float(mac_table["MAC_locator_restricted_seed_p_block"])
    representation_suspect = bool(
        (math.isfinite(locator_seed_mac) and locator_seed_mac < 0.95)
        or (
            math.isfinite(mac_seed_recovered)
            and mac_seed_recovered >= 0.9
            and math.isfinite(reported_mac)
            and reported_mac < 0.01
        )
    )

    verdict = _assign_verdict(
        continuation=continuation,
        mac_seed_recovered=mac_seed_recovered,
        d_frac=d_frac,
        rel_res=rel_res,
        mac_locator_seed=locator_seed_mac,
        representation_suspect=representation_suspect,
    )

    audit = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": CASE_ID,
        "seed_rayleigh_f_hz": TARGET_F_HZ,
        "recovered_mode_identification": recovered_entry,
        "layout_validation": layout_validation,
        "pressure_vector_stats": pressure_vectors,
        "pressure_mac_table": mac_table,
        "pressure_mac_mass_weighted": mass_weighted_mac,
        "pressure_mac_via_p_to_W_helpers": mac_via_p_to_W,
        "candidate_mode_enumeration": candidate_rows,
        "selection_audit": selection_audit,
        "replay": {
            "seed_mode_relative_residual": seed_replay.get("relative_residual"),
            "seed_mode_rayleigh_frequency_hz": seed_replay.get("rayleigh_frequency_hz"),
            "recovered_mode_relative_residual": recovered_replay.get("relative_residual"),
            "recovered_mode_rayleigh_frequency_hz": recovered_replay.get("rayleigh_frequency_hz"),
        },
        "prior_diagnostic_evaluation_excerpt": {
            "diagnostic_verdict": prior_eval.get("diagnostic_verdict"),
            "reported_MAC": reported_mac,
            "reported_replay_residual": prior_eval.get("recovered_mode", {}).get(
                "replay_relative_residual_of_recovered_mode"
            ),
        },
        "interpretation": {
            "prior_MAC_likely_used_wrong_metric_or_ordering": bool(
                math.isfinite(reported_mac)
                and reported_mac < 0.01
                and math.isfinite(mac_seed_recovered)
                and mac_seed_recovered >= MAC_TOL
            ),
            "locator_and_seed_pressure_blocks_aligned": bool(
                math.isfinite(locator_seed_mac) and locator_seed_mac >= 0.99
            ),
        },
        "audit_verdict": verdict,
        "corrected_baseline_verdict": verdict,
        "mesh_convergence_may_resume": False,
        "staged_status": {
            "mesh_convergence_pass": "Pending",
            "v2_production_promotion_ready": False,
            "lhs_promotion_blocked": True,
        },
    }

    write_json(AUDIT_JSON, audit)

    lines = [
        "# L_mid seed-branch recovery MAC / representation audit (baseline)",
        "",
        f"Generated: {audit['generated_utc']}",
        "",
        f"**Corrected baseline verdict:** `{verdict}`",
        "",
        "## Recovered mode file",
        "",
        f"- File: `{recovered_entry.get('recovered_mode_file')}`",
        f"- Index: {recovered_entry.get('recovered_mode_index')}",
        f"- f: {recovered_entry.get('recovered_mode_frequency_hz')} Hz",
        f"- vector_length: {recovered_entry.get('vector_length')}",
        "",
        "## Pressure MAC (active-pressure block, same ordering)",
        "",
        f"- MAC(locator_restricted, seed_p_block): {mac_table['MAC_locator_restricted_seed_p_block']:.6f}",
        f"- MAC(locator_restricted, recovered_p_block): {mac_table['MAC_locator_restricted_recovered_p_block']:.6f}",
        f"- MAC(seed_p_block, recovered_p_block): {mac_table['MAC_seed_p_block_recovered_p_block']:.6f}",
        "",
        "## Replay (physical-coupling-enabled operator)",
        "",
        f"- seed relative_residual: {seed_replay.get('relative_residual')}",
        f"- seed rayleigh_f_hz: {seed_replay.get('rayleigh_frequency_hz')}",
        f"- recovered relative_residual: {recovered_replay.get('relative_residual')}",
        f"- recovered rayleigh_f_hz: {recovered_replay.get('rayleigh_frequency_hz')}",
        "",
        "## Prior vs corrected MAC",
        "",
        f"- Prior reported MAC: {reported_mac}",
        f"- Corrected MAC(seed, recovered) at same f: {mac_seed_recovered:.6f}",
        "",
        f"**mesh_convergence_may_resume:** `False`",
        "",
    ]
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if DIAG_REPORT_JSON.is_file():
        diag_report = json.loads(DIAG_REPORT_JSON.read_text(encoding="utf-8"))
        row = diag_report.setdefault("baseline_coupled_v2", {})
        ev = row.setdefault("evaluation", {})
        ev["diagnostic_verdict"] = verdict
        ev["corrected_after_mac_audit"] = True
        ev["mac_audit_json"] = str(AUDIT_JSON)
        ev["pressure_MAC_to_true_acoustic_reference_corrected"] = mac_seed_recovered
        ev["replay_relative_residual_of_recovered_mode"] = recovered_replay.get("relative_residual")
        ev["replay_rayleigh_f_hz"] = recovered_replay.get("rayleigh_frequency_hz")
        row["diagnostic_verdict"] = verdict
        diag_report["mac_audit_corrected_verdict"] = verdict
        diag_report["mesh_convergence_may_resume"] = False
        write_json(DIAG_REPORT_JSON, diag_report)

    if DIAG_REPORT_MD.is_file():
        text = DIAG_REPORT_MD.read_text(encoding="utf-8")
        if "Corrected verdict (post MAC audit)" not in text:
            DIAG_REPORT_MD.write_text(
                text
                + f"\n## Corrected verdict (post MAC audit)\n\n**`{verdict}`**\n\n"
                f"MAC(seed_p, recovered_p)={mac_seed_recovered:.6f}; "
                f"replay_residual={recovered_replay.get('relative_residual')}\n",
                encoding="utf-8",
            )

    print(f"[mac_audit] verdict={verdict} MAC_seed_rec={mac_seed_recovered:.6f}", flush=True)
    print(f"[mac_audit] wrote {AUDIT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
