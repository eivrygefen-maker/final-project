#!/usr/bin/env python3
"""
Isolated M4 geometry/audio validation (experimental — does not touch production pipeline).

Test A: operator provenance audit (no solve).
Test B: aperture mic-proxy recompute + narrow-band solve plan for two extreme samples.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_DIR = SCRIPT_DIR.parent / "docs"
VALIDATION_ROOT_REL = Path(
    "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/validation/mic_proxy_v1"
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_aperture_pressure_mask import (  # noqa: E402
    aperture_mask_summary,
    build_aperture_pressure_mask,
    write_aperture_mask_npz,
)
from v2_b3_m4_lhs_pool_bridge import DEFAULT_RUN_ID_SUFFIX, lhs_entry_index, load_lhs_pool  # noqa: E402
from v2_b3_m4_lprod_interfaces import extract_geometry_dict  # noqa: E402
from v2_b3_m4_rom_fom_compare_lib import load_fom_modes_catalog_deduped  # noqa: E402
from v2_b3_m4_worker_run_lib import detect_repo_root, load_json, rel  # noqa: E402
from v2_b3_mode_audio_coupling_experimental import compute_experimental_audio_coupling  # noqa: E402
from v2_b3_rich_modal_lib import load_region_dof_bundle, prolongate_active_to_W  # noqa: E402

DEFAULT_LHS = "ROM/classic/lhs_pool.json"
BANDS = ((270.0, 290.0, "band_281"), (380.0, 400.0, "band_390"))
NARROW_TARGETS_281 = [272.0, 278.0, 281.5, 286.0]
NARROW_TARGETS_390 = [382.0, 388.0, 391.5, 396.0]


def _parse_samples(arg: str) -> List[str]:
    return [p.strip() for p in arg.split(",") if p.strip()]


def _estimate_cavity_volume(geom: Mapping[str, float]) -> float:
    length = float(geom["length"])
    width = float(geom["width"])
    depth = float(geom["depth"])
    top_t = float(geom.get("top_thickness") or 0.003)
    return length * width * max(depth - 2.0 * top_t, 1e-6)


def pick_extreme_lhs_samples(
    pool: Mapping[str, Any],
    *,
    max_index: int = 35,
    require_run_id: bool = False,
) -> Tuple[str, str]:
    """Pick smallest/largest cavity-volume among sample_000..sample_{max_index}."""
    scored: List[Tuple[float, str]] = []
    for entry in pool.get("entries") or []:
        sid = str(entry.get("id") or "")
        if not sid.startswith("sample_"):
            continue
        try:
            idx = int(sid.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if idx > int(max_index):
            continue
        if require_run_id and not entry.get("last_run_id"):
            continue
        geom = extract_geometry_dict(entry)
        if not geom:
            continue
        scored.append((_estimate_cavity_volume(geom), sid))
    if len(scored) < 2:
        return "sample_001", "sample_034"
    scored.sort(key=lambda x: x[0])
    return scored[0][1], scored[-1][1]


def _run_root(repo_root: Path, sample_id: str, pool: Mapping[str, Any], run_id_suffix: str) -> Path:
    idx = lhs_entry_index(pool, sample_id)
    entry = (pool.get("entries") or [])[idx] if idx is not None else {}
    run_id = str(entry.get("last_run_id") or f"{sample_id}_{run_id_suffix}")
    return (
        repo_root
        / "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars"
        / sample_id
        / "runs"
        / run_id
    )


def _modes_in_band(modes: Sequence[Mapping[str, Any]], lo: float, hi: float) -> List[Dict[str, Any]]:
    out = []
    for m in modes:
        f = m.get("frequency_hz")
        if f is None:
            continue
        try:
            fv = float(f)
        except (TypeError, ValueError):
            continue
        if lo <= fv <= hi:
            out.append(dict(m))
    out.sort(key=lambda r: float(r["frequency_hz"]))
    return out


def _try_load_mode_vector(worker_dir: Path, mode_index: int) -> Optional[np.ndarray]:
    for name in ("mode_vector_active.npy", "x_active.npy"):
        p = worker_dir / name
        if p.is_file():
            return np.load(p)
    rich = worker_dir.parent.parent / "lprod" / "checkpoint" / "rich_modal" / "modes_active.npz"
    if rich.is_file():
        z = np.load(rich, allow_pickle=False)
        if "vectors" in z.files:
            vecs = z["vectors"]
            if mode_index < len(vecs):
                return np.asarray(vecs[mode_index], dtype=np.float64)
    return None


def test_a_provenance(repo_root: Path, samples: Sequence[str]) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "audit_m4_operator_provenance.py"),
        "--samples",
        ",".join(samples),
        "--dolfinx",
        "--json-out",
        str(DOCS_DIR / "M4_OPERATOR_PROVENANCE_AUDIT.json"),
    ]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr[-4000:] if proc.stderr else "",
    }


def test_b_proxy_recompute(
    *,
    repo_root: Path,
    pool: Mapping[str, Any],
    sample_id: str,
    run_id_suffix: str,
    use_sample_mesh: bool,
) -> Dict[str, Any]:
    run_root = _run_root(repo_root, sample_id, pool, run_id_suffix)
    ckpt = run_root / "lprod" / "checkpoint"
    built_path = ckpt / "built_metadata.json"
    catalog_path = run_root / "aggregation" / "modes_catalog.jsonl"
    out: Dict[str, Any] = {"sample_id": sample_id, "run_root": rel(run_root, repo_root=repo_root)}

    if not built_path.is_file() or not catalog_path.is_file():
        out["status"] = "missing_checkpoint_or_catalog"
        return out

    built = load_json(built_path)
    geom = extract_geometry_dict((pool.get("entries") or [])[lhs_entry_index(pool, sample_id) or 0])
    mesh_file = run_root / "lprod" / "mesh" / "L_prod" / f"{sample_id}.msh"
    if not use_sample_mesh or not mesh_file.is_file():
        from v2_mesh_convergence_common import mesh_path  # noqa: WPS433

        mesh_file = mesh_path("L_prod", "baseline_coupled_v2")

    try:
        mask = build_aperture_pressure_mask(mesh_file, geometry=geom, built_meta=built)
        val_dir = repo_root / VALIDATION_ROOT_REL / sample_id
        write_aperture_mask_npz(val_dir / "aperture_pressure_mask.npz", mask)
        out["aperture_mask"] = aperture_mask_summary(mask)
    except Exception as exc:  # noqa: BLE001
        out["aperture_mask_error"] = f"{type(exc).__name__}:{exc}"
        return out

    region_ctx = load_region_dof_bundle(ckpt, built)
    raw_modes, deduped, _ = load_fom_modes_catalog_deduped(catalog_path)
    comparisons: List[Dict[str, Any]] = []

    for lo, hi, band_name in BANDS:
        for m in _modes_in_band(deduped, lo, hi)[:4]:
            rec = {
                "band": band_name,
                "frequency_hz": m.get("frequency_hz"),
                "legacy_mic_output_proxy": m.get("mic_output_proxy"),
                "legacy_mic_output_method": m.get("mic_output_method"),
                "air_share": m.get("air_share"),
            }
            chunk_id = m.get("chunk_id")
            x_active = None
            if chunk_id:
                wr = run_root / "worker_results" / str(chunk_id)
                x_active = _try_load_mode_vector(wr, int(m.get("mode_index") or 0))
            if x_active is not None:
                built_operators = {"active_local": np.asarray(built["active_local"], dtype=np.int32)}
                # Minimal built dict for prolongation — full solve uses richer built_from_checkpoint_metadata.
                try:
                    from v2_b3_operator_checkpoint_portable import built_from_checkpoint_metadata  # noqa: WPS433
                    from v2_b3_petsc_util import load_operators_with_portable_fallback  # noqa: WPS433

                    A, M, _ = load_operators_with_portable_fallback(ckpt)
                    built_full, _ = built_from_checkpoint_metadata(built, A_active=A, M_active=M)
                    exp = compute_experimental_audio_coupling(
                        x_active=x_active,
                        built=built_full,
                        region_ctx=region_ctx,
                        p_idx_aperture=mask["p_idx_aperture"],
                    )
                    rec.update(
                        {
                            "experimental_mic_output_proxy": exp.get("mic_output_proxy"),
                            "experimental_mic_output_method": exp.get("mic_output_method"),
                            "experimental_mic_ratio_vs_legacy": exp.get("experimental_mic_ratio_vs_legacy"),
                            "recompute_status": "from_mode_vector",
                        }
                    )
                    A.destroy()
                    M.destroy()
                except Exception as exc:  # noqa: BLE001
                    rec["recompute_status"] = f"vector_recompute_failed:{type(exc).__name__}:{exc}"
            else:
                rec["recompute_status"] = "mode_vector_not_retained; narrow_band_solve_required"
            comparisons.append(rec)

    out["band_comparisons"] = comparisons
    out["status"] = "ok"
    return out


def _narrow_band_commands(repo_root: Path, sample_id: str) -> List[str]:
    val_dir = repo_root / VALIDATION_ROOT_REL / sample_id
    targets_path = val_dir / "narrow_band_targets.json"
    targets_doc = {
        "schema": "m4_validation_narrow_band_targets_v1",
        "sample_id": sample_id,
        "bands_hz": [{"name": "281", "targets_hz": NARROW_TARGETS_281}, {"name": "390", "targets_hz": NARROW_TARGETS_390}],
        "experimental_env": {
            "B3_MIC_PROXY_MODE": "aperture_pressure_rms_v1",
            "B3_EXPERIMENTAL_APERTURE_MASK_NPZ": str(val_dir / "aperture_pressure_mask.npz"),
        },
    }
    val_dir.mkdir(parents=True, exist_ok=True)
    targets_path.write_text(json.dumps(targets_doc, indent=2) + "\n", encoding="utf-8")

    solve_script = SCRIPT_DIR / "v2_b3_checkpoint_solve_target_list.py"
    return [
        (
            f"export B3_MIC_PROXY_MODE=aperture_pressure_rms_v1\n"
            f"export B3_EXPERIMENTAL_APERTURE_MASK_NPZ={val_dir / 'aperture_pressure_mask.npz'}\n"
            f"python {solve_script} \\\n"
            f"  --checkpoint pipeline_runs/guitars/{sample_id}/runs/<run_id>/lprod/checkpoint \\\n"
            f"  --targets-json {targets_path} \\\n"
            f"  --output-dir {val_dir / 'narrow_band_solve'}"
        )
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="M4 geometry/audio validation (experimental, isolated).")
    parser.add_argument("--lhs-json", type=Path, default=Path(DEFAULT_LHS))
    parser.add_argument("--samples", default="", help="Two samples; default = LHS volume extremes.")
    parser.add_argument("--run-id-suffix", default=DEFAULT_RUN_ID_SUFFIX)
    parser.add_argument("--test-a-only", action="store_true")
    parser.add_argument("--test-b-only", action="store_true")
    parser.add_argument("--use-sample-mesh-for-mask", action="store_true", help="Build aperture mask on sample Gmsh mesh.")
    parser.add_argument(
        "--max-sample-index",
        type=int,
        default=35,
        help="When auto-picking extremes, only consider sample_000..sample_N (default 35).",
    )
    parser.add_argument("--json-out", type=Path, default=DOCS_DIR / "M4_GEOMETRY_AUDIO_VALIDATION.json")
    args = parser.parse_args(argv)

    repo_root = detect_repo_root(SCRIPT_DIR)
    pool = load_lhs_pool(args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json)

    if args.samples:
        samples = _parse_samples(args.samples)
        if len(samples) < 2:
            samples = samples + [pick_extreme_lhs_samples(pool)[1]]
    else:
        samples = list(pick_extreme_lhs_samples(pool, max_index=int(args.max_sample_index)))

    report: Dict[str, Any] = {
        "schema": "m4_geometry_audio_validation_v1",
        "samples": samples,
        "validation_root": str(VALIDATION_ROOT_REL),
        "production_decision_recommendation": "RERUN_ALL_35_SAMPLES",
        "decision_rationale": (
            "Operator assembly uses fixed baseline_coupled_v2 topology; LHS geometry in generated "
            "Gmsh meshes does not affect eigenvalue problem coordinates. Frequencies are valid only "
            "for the material-overlay-on-fixed-topology model, not for per-sample body geometry. "
            "Mic proxy used cavity_pressure_max with empty soundhole mask — audio fields require recompute. "
            "Full rerun required after wiring sample mesh into Stage A and aperture pressure proxy."
        ),
    }

    if not args.test_b_only:
        report["test_a_provenance"] = test_a_provenance(repo_root, samples)
        print(report["test_a_provenance"].get("stdout", ""))

    if not args.test_a_only:
        report["test_b_samples"] = [
            test_b_proxy_recompute(
                repo_root=repo_root,
                pool=pool,
                sample_id=sid,
                run_id_suffix=str(args.run_id_suffix),
                use_sample_mesh=bool(args.use_sample_mesh_for_mask),
            )
            for sid in samples[:2]
        ]
        report["test_b_narrow_band_commands"] = {
            sid: _narrow_band_commands(repo_root, sid) for sid in samples[:2]
        }

    out = args.json_out if args.json_out.is_absolute() else repo_root / args.json_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"validation_report={rel(out, repo_root=repo_root)}")
    print(f"recommended_decision={report['production_decision_recommendation']}")
    for sid in samples[:2]:
        print(f"  extreme_sample={sid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
