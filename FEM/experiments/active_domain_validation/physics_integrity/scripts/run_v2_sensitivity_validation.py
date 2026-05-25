#!/usr/bin/env python3
"""
Experiment-only v2 sensitivity validation around frozen coupled_physical_core_v2.

Pilot: soundhole radius small/large only. Full suite: manifest samples (after pilot passes).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
PHYSICS_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PHYSICS_ROOT / "scripts"
if str(REPO_ROOT / "FEM" / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "FEM" / "scripts"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fem_mode_array_utils import load_mode_column_any
from v2_sensitivity_gates import run_mesh_gates
from v2_sensitivity_mesh import build_sample_mesh, sample_geometry, sample_mesh_path

SOLVE_SCRIPT = SCRIPT_DIR / "v2_sensitivity_solve.py"


def _hz_result_tag(hz: float) -> int:
    return int(round(float(hz) * 1000))


def _pressure_subspace_mac(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    p_to_W: np.ndarray,
    *,
    scale_p_a: float = 1.0,
    scale_p_b: float = 1.0,
) -> float:
    p_idx = np.asarray(p_to_W, dtype=np.int32).ravel()
    if p_idx.size == 0:
        return float("nan")
    pa = np.asarray(vec_a[p_idx], dtype=np.complex128).ravel() * float(scale_p_a)
    pb = np.asarray(vec_b[p_idx], dtype=np.complex128).ravel() * float(scale_p_b)
    na = float(np.linalg.norm(pa))
    nb = float(np.linalg.norm(pb))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float(abs(np.vdot(pa, pb)) / (na * nb))

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


def _load_mode_dense(path: Path, n_coupled_W: int) -> np.ndarray:
    col = load_mode_column_any(path)
    dense = np.asarray(col.toarray(), dtype=np.float64).ravel()
    if dense.size != int(n_coupled_W):
        raise ValueError(f"mode length {dense.size} != n_coupled_W {n_coupled_W} in {path}")
    return dense


def _run_mpi_v2_solve(
    sample: Dict[str, Any],
    mesh_path: Path,
    *,
    target_hz: float,
    log_path: Path,
) -> Tuple[int, Dict[str, Any]]:
    """Launch v2 solve under mpiexec -n 1 (single MPI child per sample)."""
    sample_id = str(sample["id"])
    case_dir = SENS_ROOT / "samples" / sample_id
    case_dir.mkdir(parents=True, exist_ok=True)
    sample_json = case_dir / "sample_spec.json"
    sample_json.write_text(json.dumps(sample, indent=2), encoding="utf-8")
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
        str(float(target_hz)),
    ]
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result_path = case_dir / "results" / f"result_{_hz_result_tag(target_hz)}.json"
    if not result_path.is_file():
        return int(proc.returncode), {
            "error": f"solve worker exit {proc.returncode}; missing {result_path}",
            "solve_log": str(log_path),
            "mpi_command": " ".join(cmd),
        }
    solve = json.loads(result_path.read_text(encoding="utf-8"))
    solve["solve_exit_code"] = int(proc.returncode)
    solve["solve_log"] = str(log_path)
    return int(proc.returncode), solve


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
        (case_dir / "results" / f"result_{_hz_result_tag(target_hz)}.json").read_text(
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
    hz_tag = _hz_result_tag(target_hz)
    mode_files = sorted((case_dir / "modes").glob(f"mode_{hz_tag}_*.smx.npz"))
    if not mode_files:
        mode_files = sorted((case_dir / "modes").glob("mode_*.smx.npz"))
    if not mode_files:
        return None
    diag_path = case_dir / "diagnostics" / "mode_physics_diagnostics.json"
    meta: List[Dict[str, Any]] = []
    if diag_path.is_file():
        meta = list(json.loads(diag_path.read_text(encoding="utf-8")).get("modes") or [])
    row = next(
        (m for m in meta if abs(float(m.get("frequency_hz", 0)) - ref_hz) < 0.05),
        None,
    )
    if row and row.get("vector_path"):
        ref_path = (case_dir / str(row["vector_path"])).resolve()
    else:
        ref_path = mode_files[0]
        row = row or {}
    prior_path = case_dir / "results" / f"result_{hz_tag}.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
    n_W = int(prior.get("n_reduced_W", 112100))
    vec = _load_mode_dense(ref_path, n_W)
    p_to_W = np.asarray(prior.get("p_to_W") or [], dtype=np.int32).ravel()
    if p_to_W.size == 0:
        coupled_prior_path = (
            V2_ROOT / str(frozen["subcase_coupled"]) / "results" / f"result_{hz_tag}.json"
        )
        if coupled_prior_path.is_file():
            coupled_prior = json.loads(coupled_prior_path.read_text(encoding="utf-8"))
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
    mac = _pressure_subspace_mac(
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

    if sample.get("requires_remesh"):
        try:
            mesh_path = build_sample_mesh(sample)
            row["mesh_built"] = True
        except Exception as exc:
            row["status"] = "mesh_build_failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            return row
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
        row["error"] = (
            "combined_mesh_gate_pass=False "
            f"(aperture_exit={gates.get('aperture_audit_exit')} "
            f"adjacency_exit={gates.get('adjacency_audit_exit')})"
        )
        return row

    if skip_solve:
        row["status"] = "gates_only"
        return row

    log_path = case_dir / "logs" / "v2_solve.log"
    rc, solve = _run_mpi_v2_solve(sample, mesh_path, target_hz=target_hz, log_path=log_path)
    if rc != 0 or not solve.get("v2_converged"):
        row.update(solve)
        row["status"] = "solve_failed"
        row["error"] = solve.get("error") or f"mpi solve exit {rc}"
        return row

    row.update(solve)
    row["status"] = "ok"
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
        cand_vec = _load_mode_dense(cand_path, n_W)
        gnhep_full = dict(solve.get("gnhep_scales") or {})
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

    manifest = _load_manifest()
    target_hz = float(
        json.loads(V2_CONFIG.read_text(encoding="utf-8"))
        .get("solver", {})
        .get("_worker_target_hz", 244.39)
    )

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
        print(f"[v2_sensitivity] sample={sid}", flush=True)
        row = _process_sample(
            sample,
            {},
            manifest,
            baseline_ref,
            target_hz=target_hz,
            skip_solve=args.gates_only,
        )
        results[sid] = row
        _write_json(DIAG_DIR / "v2_sensitivity_validation_summary.partial.json", {"samples": results})

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
        f_a = row.get(
            "nearest_acoustic_f_hz",
            (row.get("nearest_acoustic_branch") or {}).get("frequency_hz", float("nan")),
        )
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
    failed = [
        sid
        for sid, r in results.items()
        if not r.get("ingest_only") and r.get("status") != "ok"
    ]
    return 0 if not failed else 4


if __name__ == "__main__":
    raise SystemExit(main())
