#!/usr/bin/env python3
"""
Experiment-only v2 sensitivity validation around frozen coupled_physical_core_v2.

Pilot: soundhole radius small/large only. Full suite: manifest samples (after pilot passes).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "FEM" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PHYSICS_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PHYSICS_ROOT / "scripts"))

import fem_main_3d as fem3d
from coupled_participation_audit import _load_coupled_mode_dense_vector, _load_modes
from fem_mode_array_utils import MODE_VECTOR_FILE_SUFFIX, dense_to_csr_f32_column, save_mode_csr
from fem_worker_single import _apply_master_worker_solver_profile, hz_result_tag
from mode_diagnostics import (
    compute_mass_energy_participation,
    diagnose_mixed_mode,
    merge_scaling_metadata,
    pressure_subspace_mac,
)
from v2_sensitivity_gates import run_mesh_gates
from v2_sensitivity_mesh import (
    NOMINAL_GEOMETRY,
    build_sample_mesh,
    sample_geometry,
    sample_mesh_path,
)

V2_ROOT = PHYSICS_ROOT / "coupled_physical_core_v2"
V2_CONFIG = PHYSICS_ROOT / "configs" / "coupled_physical_core_v2.json"
MANIFEST = PHYSICS_ROOT / "configs" / "v2_sensitivity_manifest.json"
SENS_ROOT = PHYSICS_ROOT / "v2_sensitivity_validation"
DIAG_DIR = SENS_ROOT / "diagnostics"
VALIDATION_MESH = (EXPERIMENT_ROOT / "mesh" / "validation_tiny_guitar_3d.msh").resolve()

BAND_LO = 220.0
BAND_HI = 265.0
BASELINE_F_HZ = 244.39159990162557
FREQ_NOISE_HZ = 0.01
ENERGY_ACOUSTIC_THRESHOLD = 0.85


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_manifest() -> Dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _classify_phys_energy(p_frac: float) -> str:
    if float(p_frac) >= ENERGY_ACOUSTIC_THRESHOLD:
        return "acoustic_dominated"
    if float(p_frac) <= 0.15:
        return "structural_dominated"
    return "mixed"


def _resolve_mesh_path(sample: Dict[str, Any]) -> Path:
    if sample.get("ingest_only"):
        return VALIDATION_MESH
    if sample.get("requires_remesh"):
        return sample_mesh_path(str(sample["id"]))
    return VALIDATION_MESH


def _apply_material_overrides(cfg: dict, sample: Dict[str, Any]) -> None:
    overrides = sample.get("materials_override") or {}
    top = overrides.get("top") or {}
    scale = float(top.get("E_L_scale", 1.0))
    if abs(scale - 1.0) > 1.0e-12:
        mat = cfg.setdefault("materials", {}).setdefault("top", {})
        mat["E_L"] = float(mat.get("E_L", 0.0)) * scale


def _solve_v2_sample(
    cfg_base: dict,
    mesh_path: Path,
    *,
    sample_id: str,
    target_hz: float,
) -> Dict[str, Any]:
    case_dir = SENS_ROOT / "samples" / sample_id
    sorting = case_dir / "sorting"
    for d in (sorting, case_dir / "logs", case_dir / "modes", case_dir / "diagnostics"):
        d.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting.resolve())

    cfg = copy.deepcopy(cfg_base)
    sc = cfg.setdefault("solver", {})
    sc["mesh_file"] = str(mesh_path.resolve())
    sc["coupled_physical_core_v2_diagnosis"] = True
    sc["coupled_physical_core_v2_coupling_enabled"] = True
    sc["fsi_coupling_gain"] = 1.0
    sc["fsi_nitsche_enable"] = False
    sc["physics_integrity_capture"] = True
    sc["coupled_air_pressure_restriction_diagnosis"] = True
    sc["physics_integrity_branch"] = f"v2-sensitivity-{sample_id}"
    sc["_worker_target_hz"] = target_hz
    sc["_worker_harvest_lo_hz"] = BAND_LO
    sc["_worker_harvest_hi_hz"] = BAND_HI

    eps_band_solver = str(sc.get("eps_band_solver", "shift_invert")).strip() or "shift_invert"
    nm = _apply_master_worker_solver_profile(
        cfg,
        num_modes=int(sc.get("num_modes", 12)),
        structural_only=False,
        eps_band_solver=eps_band_solver,
    )
    lam_t = (2.0 * math.pi * target_hz) ** 2
    sc["_worker_eps_target_lambda"] = lam_t
    cfg["_worker_target_hz"] = target_hz
    cfg["_worker_num_modes"] = nm
    cfg["geometry"] = sample_geometry(sample)

    t0 = time.perf_counter()
    _msh, _W, freqs_hz, eigvecs, _nu, _np = fem3d._solve_coupled_evp(
        mesh_file=mesh_path,
        config=cfg,
        num_modes=nm,
    )
    elapsed = time.perf_counter() - t0

    restr = cfg.get("_coupled_air_pressure_restriction") or {}
    u_to_W = np.asarray(cfg["_coupled_air_u_to_W_map"], dtype=np.int32).ravel()
    p_to_W = np.asarray(cfg["_coupled_air_p_to_W_map"], dtype=np.int32).ravel()
    gnhep = merge_scaling_metadata(case_dir)
    pi = cfg.get("_physics_integrity") or {}
    if isinstance(pi, dict) and pi.get("gnhep_scales"):
        gnhep.update({k: float(v) for k, v in pi["gnhep_scales"].items()})

    # Replay A/M for physical energy (no second eigensolve).
    cfg_am = copy.deepcopy(cfg)
    cfg_am.setdefault("solver", {})["coupled_air_pressure_restriction_replay_audit"] = True
    sorting_am = case_dir / "sorting_energy"
    sorting_am.mkdir(parents=True, exist_ok=True)
    fem3d.set_sorting_root(sorting_am.resolve())
    _m2, _W2, A, M = fem3d._solve_coupled_evp(
        mesh_file=mesh_path,
        config=cfg_am,
        num_modes=0,
        solve_evp=False,
    )
    fem3d.set_sorting_root(sorting.resolve())

    hz_tag = hz_result_tag(target_hz)
    mode_rows: List[Dict[str, Any]] = []
    in_band: List[Dict[str, Any]] = []
    n_modes = int(eigvecs.shape[1]) if eigvecs.ndim == 2 else 0
    for j in range(n_modes):
        vec = eigvecs[:, j]
        mode_path = case_dir / "modes" / f"mode_{hz_tag}_{j:03d}{MODE_VECTOR_FILE_SUFFIX}"
        save_mode_csr(mode_path, dense_to_csr_f32_column(vec))
        diag = diagnose_mixed_mode(
            vec, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep, frequency_hz=float(freqs_hz[j])
        )
        energy = compute_mass_energy_participation(
            vec, M, A, u_to_W=u_to_W, p_to_W=p_to_W, gnhep=gnhep
        )
        row = {
            **diag,
            **{k: energy[k] for k in energy if k.endswith("_phys") or k == "p_frac_energy_phys"},
            "mode_index": j,
            "frequency_hz": float(freqs_hz[j]),
            "vector_path": str(mode_path.relative_to(case_dir)).replace("\\", "/"),
            "mode_class_physical_energy": _classify_phys_energy(
                float(energy["p_frac_energy_phys"])
            ),
        }
        mode_rows.append(row)
        if BAND_LO <= float(freqs_hz[j]) <= BAND_HI:
            in_band.append(row)

    try:
        A.destroy()
        M.destroy()
    except Exception:
        pass

    eps_diag = cfg.get("_eps_batch_diagnostics") or sc.get("_eps_batch_diagnostics") or {}
    ref_hz = BASELINE_F_HZ
    acoustic_pool = [
        m
        for m in in_band
        if m["mode_class_physical_energy"] == "acoustic_dominated"
        or float(m["p_frac_energy_phys"]) >= 0.35
    ]
    pool = acoustic_pool if acoustic_pool else in_band
    nearest = (
        min(pool, key=lambda m: abs(float(m["frequency_hz"]) - ref_hz)) if pool else None
    )

    result = {
        "sample_id": sample_id,
        "elapsed_s": elapsed,
        "mesh_file": str(mesh_path),
        "n_reduced_W": int(restr.get("n_reduced_W", -1)),
        "n_u_active": int(restr.get("n_u_active", u_to_W.size)),
        "n_p_active": int(restr.get("n_p_active", p_to_W.size)),
        "p_to_W": p_to_W.tolist(),
        "eps_batch_diagnostics": eps_diag,
        "nconv_marked": int(eps_diag.get("nconv_marked", -1)),
        "v2_converged": int(eps_diag.get("nconv_marked", -1)) > 0,
        "in_band_modes": in_band,
        "nearest_acoustic_branch": nearest,
        "num_modes_saved": n_modes,
        "gnhep_scales": {k: float(gnhep.get(k, 1.0)) for k in ("s_uu", "s_pp", "s_couple")},
    }
    _write_json(case_dir / "results" / f"result_{hz_tag}.json", result)
    _write_json(case_dir / "diagnostics" / "mode_energy_summary.json", {"modes": mode_rows})
    return result


def _ingest_baseline(manifest: Dict[str, Any]) -> Dict[str, Any]:
    frozen = manifest["frozen_baseline"]
    sub = str(frozen["subcase_coupled"])
    case_dir = V2_ROOT / sub
    target_hz = float(
        json.loads(V2_CONFIG.read_text(encoding="utf-8"))
        .get("solver", {})
        .get("_worker_target_hz", 244.39)
    )
    prior = json.loads(
        (case_dir / "results" / f"result_{hz_result_tag(target_hz)}.json").read_text(
            encoding="utf-8"
        )
    )
    nearest = prior.get("nearest_acoustic_mode") or {}
    in_band = prior.get("in_band_modes") or []
    return {
        "sample_id": "baseline_nominal",
        "ingest_only": True,
        "mesh_file": str(VALIDATION_MESH),
        "mesh_gates_skipped": True,
        "v2_converged": True,
        "n_p_active": int(prior.get("n_p_active", -1)),
        "nearest_acoustic_branch": nearest,
        "in_band_modes": in_band,
        "source": str(case_dir),
        "frozen_baseline_f_hz": float(frozen["acoustic_reference_f_hz"]),
    }


def _load_baseline_pressure_reference(
    manifest: Dict[str, Any],
    target_hz: float,
) -> Optional[Dict[str, Any]]:
    frozen = manifest["frozen_baseline"]
    ref_hz = float(frozen["acoustic_reference_f_hz"])
    case_dir = V2_ROOT / str(frozen["subcase_reference"])
    meta, mode_files = _load_modes(case_dir, target_hz)
    if not mode_files:
        return None
    row = next(
        (m for m in meta if abs(float(m.get("frequency_hz", 0)) - ref_hz) < 0.05),
        None,
    )
    if row and row.get("vector_path"):
        ref_path = (case_dir / str(row["vector_path"])).resolve()
    else:
        ref_path = sorted((case_dir / "modes").glob("mode_*.smx.npz"))[0]
        row = meta[0] if meta else {}
    prior_path = V2_ROOT / str(frozen["subcase_reference"]) / "results" / f"result_{hz_result_tag(target_hz)}.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
    n_W = int(prior.get("n_reduced_W", 112100))
    vec, _ = _load_coupled_mode_dense_vector(
        ref_path, n_coupled_W=n_W, mode_index=int(row.get("mode_index", 0))
    )
    p_to_W = np.asarray(prior.get("p_to_W") or [], dtype=np.int32).ravel()
    if p_to_W.size == 0:
        coupled_prior = json.loads(
            (
                V2_ROOT
                / str(frozen["subcase_coupled"])
                / "results"
                / f"result_{hz_result_tag(target_hz)}.json"
            ).read_text(encoding="utf-8")
        )
        p_to_W = np.asarray(coupled_prior.get("p_to_W") or [], dtype=np.int32).ravel()
    return {
        "frequency_hz": float(row.get("frequency_hz", ref_hz)),
        "vector": vec,
        "p_to_W": p_to_W,
        "n_p_active_baseline": int(prior.get("n_p_active", p_to_W.size)),
        "vector_path": str(ref_path),
    }


def _mac_to_baseline(
    cand_vec: np.ndarray,
    cand_p_to_W: np.ndarray,
    cand_n_p: int,
    baseline_ref: Dict[str, Any],
    gnhep: Dict[str, float],
) -> Dict[str, Any]:
    n_base = int(baseline_ref.get("n_p_active_baseline", -1))
    if int(cand_n_p) != n_base or n_base <= 0:
        return {
            "mac_pressure_gnhep_undo_s_pp": None,
            "mac_comparable": False,
            "reason": f"n_p_active candidate={cand_n_p} baseline={n_base} (mesh DOF layout differs)",
        }
    ref_vec = baseline_ref["vector"]
    p_ref = np.asarray(baseline_ref.get("p_to_W") or [], dtype=np.int32).ravel()
    if ref_vec.size != cand_vec.size:
        return {
            "mac_pressure_gnhep_undo_s_pp": None,
            "mac_comparable": False,
            "reason": f"reduced vector length mismatch {cand_vec.size} vs {ref_vec.size}",
        }
    if p_ref.size != cand_p_to_W.size:
        return {
            "mac_pressure_gnhep_undo_s_pp": None,
            "mac_comparable": False,
            "reason": f"p_to_W length mismatch {cand_p_to_W.size} vs baseline {p_ref.size}",
        }
    if not np.array_equal(p_ref, cand_p_to_W):
        return {
            "mac_pressure_gnhep_undo_s_pp": None,
            "mac_comparable": False,
            "reason": "p_to_W index map differs from baseline (remeshed geometry)",
        }
    s_p = max(float(gnhep.get("s_pp", 1.0)), 1.0e-30)
    mac = pressure_subspace_mac(
        ref_vec, cand_vec, cand_p_to_W, scale_p_a=s_p, scale_p_b=s_p
    )
    return {
        "mac_pressure_gnhep_undo_s_pp": float(mac),
        "mac_comparable": True,
        "inner_product": "np.vdot on active-pressure DOFs via p_to_W",
    }


def _evaluate_expected_direction(
    sample: Dict[str, Any],
    result: Dict[str, Any],
    *,
    peer_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    exp = sample.get("expected_direction") or {}
    nearest = result.get("nearest_acoustic_branch") or {}
    f_hz = float(nearest.get("frequency_hz", float("nan")))
    delta = f_hz - BASELINE_F_HZ if math.isfinite(f_hz) else float("nan")
    out: Dict[str, Any] = {
        "parameter": exp.get("parameter"),
        "expect": exp.get("expect"),
        "delta_f_hz_from_baseline_acoustic": delta,
        "recorded": math.isfinite(delta),
    }
    param = str(exp.get("parameter", ""))
    if param == "hole_radius" and len(peer_results) >= 2:
        f_small = float(
            (peer_results.get("hole_radius_small", {}).get("nearest_acoustic_branch") or {}).get(
                "frequency_hz", float("nan")
            )
        )
        f_large = float(
            (peer_results.get("hole_radius_large", {}).get("nearest_acoustic_branch") or {}).get(
                "frequency_hz", float("nan")
            )
        )
        span = abs(f_large - f_small) if math.isfinite(f_small) and math.isfinite(f_large) else float(
            "nan"
        )
        out["hole_radius_small_f_hz"] = f_small
        out["hole_radius_large_f_hz"] = f_large
        out["span_hz_small_vs_large"] = span
        out["monotonic_trend_recorded"] = math.isfinite(span) and span > FREQ_NOISE_HZ
        if math.isfinite(f_small) and math.isfinite(f_large):
            out["trend_sign"] = "increasing" if f_large > f_small else "decreasing"
    return out


def _process_sample(
    sample: Dict[str, Any],
    cfg_base: dict,
    manifest: Dict[str, Any],
    baseline_ref: Optional[Dict[str, Any]],
    *,
    target_hz: float,
    skip_solve: bool,
) -> Dict[str, Any]:
    sample_id = str(sample["id"])
    if sample.get("ingest_only"):
        return _ingest_baseline(manifest)

    case_dir = SENS_ROOT / "samples" / sample_id
    gates_dir = case_dir / "diagnostics" / "gates"
    geom = sample_geometry(sample)
    mesh_path = _resolve_mesh_path(sample)

    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "varied_parameters": {
            k: geom[k]
            for k in ("hole_radius", "depth", "top_thickness")
            if k in geom
        },
        "materials_override": sample.get("materials_override") or {},
    }

    if sample.get("requires_remesh") and not skip_solve:
        mesh_path = build_sample_mesh(sample)
        row["mesh_built"] = True
    row["mesh_file"] = str(mesh_path)

    if not mesh_path.is_file():
        row["status"] = "failed"
        row["error"] = f"mesh missing: {mesh_path}"
        return row

    gates = run_mesh_gates(
        mesh_path,
        hole_radius_m=float(geom["hole_radius"]),
        gates_dir=gates_dir,
    )
    row["mesh_gates"] = gates
    if not gates.get("combined_mesh_gate_pass"):
        row["status"] = "mesh_gate_failed"
        return row

    if skip_solve:
        row["status"] = "gates_only"
        return row

    cfg = copy.deepcopy(cfg_base)
    cfg["geometry"] = geom
    _apply_material_overrides(cfg, sample)
    try:
        solve = _solve_v2_sample(cfg, mesh_path, sample_id=sample_id, target_hz=target_hz)
    except Exception as exc:
        row["status"] = "solve_failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    row.update(solve)
    row["status"] = "ok" if solve.get("v2_converged") else "v2_not_converged"
    nearest = solve.get("nearest_acoustic_branch") or {}
    if nearest:
        row["nearest_acoustic_f_hz"] = float(nearest["frequency_hz"])
        row["delta_f_hz_from_baseline"] = float(nearest["frequency_hz"]) - BASELINE_F_HZ
        row["p_frac_energy_phys"] = float(nearest.get("p_frac_energy_phys", float("nan")))
        row["structural_modal_energy_phys"] = float(
            nearest.get("structural_modal_energy_phys", float("nan"))
        )
        row["acoustic_modal_energy_phys"] = float(
            nearest.get("acoustic_modal_energy_phys", float("nan"))
        )
        row["mass_cross_term_phys"] = float(nearest.get("mass_cross_term_phys", float("nan")))
        row["mode_class_physical_energy"] = nearest.get("mode_class_physical_energy")

    if baseline_ref and nearest.get("vector_path"):
        cand_path = (case_dir / str(nearest["vector_path"])).resolve()
        n_W = int(solve.get("n_reduced_W", 112100))
        cand_vec, _ = _load_coupled_mode_dense_vector(
            cand_path,
            n_coupled_W=n_W,
            mode_index=int(nearest.get("mode_index", 0)),
        )
        gnhep = dict(solve.get("gnhep_scales") or {})
        gnhep_full = merge_scaling_metadata(case_dir)
        gnhep_full.update(gnhep)
        p_map = np.asarray(solve.get("p_to_W") or [], dtype=np.int32).ravel()
        row["pressure_mac_to_baseline"] = _mac_to_baseline(
            cand_vec,
            p_map,
            int(solve.get("n_p_active", p_map.size)),
            baseline_ref,
            gnhep_full,
        )

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="v2 sensitivity validation suite")
    parser.add_argument("--pilot", action="store_true", help="Run pilot samples only (hole radius)")
    parser.add_argument("--sample-id", type=str, default="", help="Run one sample by id")
    parser.add_argument("--gates-only", action="store_true", help="Build mesh + gates, skip solve")
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print("[v2_sensitivity] Requires mpiexec -n 1", file=sys.stderr)
        return 2

    manifest = _load_manifest()
    cfg_base = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    target_hz = float(cfg_base.get("solver", {}).get("_worker_target_hz", 244.39))

    pilot_ids = set(manifest.get("pilot_sample_ids") or [])
    samples = list(manifest.get("samples") or [])
    if args.pilot:
        samples = [s for s in samples if s.get("pilot") or s.get("id") == "baseline_nominal"]
    if args.sample_id:
        samples = [s for s in samples if str(s["id"]) == args.sample_id]

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    SENS_ROOT.mkdir(parents=True, exist_ok=True)

    baseline_ref = _load_baseline_pressure_reference(manifest, target_hz)
    results: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        sid = str(sample["id"])
        if sample.get("ingest_only"):
            results[sid] = results.get(sid) or _ingest_baseline(manifest)
            continue
        if MPI.COMM_WORLD.rank == 0:
            print(f"[v2_sensitivity] sample={sid}", flush=True)
        row = _process_sample(
            sample,
            cfg_base,
            manifest,
            baseline_ref,
            target_hz=target_hz,
            skip_solve=args.gates_only,
        )
        results[sid] = row

    for sid, row in list(results.items()):
        sample = next((s for s in manifest["samples"] if s["id"] == sid), None)
        if not sample or sample.get("ingest_only") or row.get("status") != "ok":
            continue
        row["expected_direction_evaluation"] = _evaluate_expected_direction(
            sample, row, peer_results=results
        )

    pilot_pass = all(
        results.get(s, {}).get("mesh_gates", {}).get("combined_mesh_gate_pass", True)
        for s in pilot_ids
        if not results.get(s, {}).get("ingest_only")
    ) and all(results.get(s, {}).get("v2_converged") for s in pilot_ids)

    hole_eval = results.get("hole_radius_small", {}).get("expected_direction_evaluation", {})
    summary = {
        "suite": "v2_sensitivity_validation",
        "frozen_baseline": manifest["frozen_baseline"],
        "pilot_mode": bool(args.pilot),
        "baseline_acoustic_f_hz": BASELINE_F_HZ,
        "samples": results,
        "promotion": {
            "lhs_blocked": True,
            "pilot_all_gates_and_v2_pass": pilot_pass if args.pilot else None,
            "hole_radius_monotonic_trend_recorded": hole_eval.get("monotonic_trend_recorded"),
            "full_suite_required_before_lhs": True,
        },
        "note": "coupled_physical_core_v2 formulation unchanged; v1 archived.",
    }
    _write_json(DIAG_DIR / "v2_sensitivity_validation_summary.json", summary)

    if MPI.COMM_WORLD.rank == 0:
        md = [
            "# v2 sensitivity validation summary",
            "",
            f"Pilot mode: `{args.pilot}`",
            "",
            "| sample | gates | v2 | f_acoustic Hz | Δf | p_frac_energy | MAC |",
            "|--------|-------|----|--------------:|---:|--------------:|----:|",
        ]
        for sid, row in results.items():
            gates = "—" if row.get("mesh_gates_skipped") else (
                "pass" if (row.get("mesh_gates") or {}).get("combined_mesh_gate_pass") else "FAIL"
            )
            v2 = "—" if row.get("ingest_only") else ("ok" if row.get("v2_converged") else "no")
            f_a = row.get("nearest_acoustic_f_hz", row.get("nearest_acoustic_branch", {}).get("frequency_hz", float("nan")))
            mac = (row.get("pressure_mac_to_baseline") or {}).get("mac_pressure_gnhep_undo_s_pp")
            mac_s = f"{mac:.4f}" if mac is not None and math.isfinite(float(mac)) else "n/a"
            md.append(
                f"| {sid} | {gates} | {v2} | {float(f_a):.6f} | "
                f"{row.get('delta_f_hz_from_baseline', float('nan')):+.6f} | "
                f"{row.get('p_frac_energy_phys', float('nan')):.4f} | {mac_s} |"
            )
        (DIAG_DIR / "v2_sensitivity_validation_summary.md").write_text(
            "\n".join(md) + "\n",
            encoding="utf-8",
        )
        print(f"[v2_sensitivity] wrote {DIAG_DIR / 'v2_sensitivity_validation_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
