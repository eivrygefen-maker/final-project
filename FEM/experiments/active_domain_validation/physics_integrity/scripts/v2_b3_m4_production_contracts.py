#!/usr/bin/env python3
"""M4 production acceptance contracts and operator-mesh provenance (geometry-corrected v1)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from v2_b3_m4_lprod_interfaces import extract_geometry_dict, geometry_fingerprint  # noqa: E402
from v2_b3_rich_modal_lib import REGION_DOF_INDICES_NPZ, load_region_dof_bundle  # noqa: E402

DATASET_VERSION = "m4_geometry_corrected_v1"
REQUIRE_APERTURE_MASK_ENV = "B3_REQUIRE_APERTURE_MASK"
ALLOW_CAVITY_MAX_FALLBACK_ENV = "B3_ALLOW_CAVITY_MAX_MIC_FALLBACK"
DIAGNOSTIC_ONLY_FALLBACK_ENV = "B3_DIAGNOSTIC_MIC_FALLBACK_ONLY"

PRODUCTION_MIC_METHOD = "aperture_pressure_rms_proxy_v1"
ALLOWED_MIC_METHODS = frozenset(
    {
        PRODUCTION_MIC_METHOD,
        "aperture_nearfield_pressure_rms_proxy_v1",
    }
)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry_from_core_config(core_config_path: Optional[Path]) -> Dict[str, float]:
    if not core_config_path or not core_config_path.is_file():
        return {}
    try:
        cfg = json.loads(core_config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return extract_geometry_dict(cfg)


def resolve_operator_mesh_file(
    *,
    operator_mesh_arg: Optional[Path],
    core_config_path: Optional[Path],
    repo_root: Path,
) -> Path:
    """Production operator mesh: explicit CLI path or core_config solver.mesh_file (no baseline fallback)."""
    if operator_mesh_arg is not None:
        path = Path(operator_mesh_arg).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"operator_mesh_file missing: {path}")
        return path
    if core_config_path and core_config_path.is_file():
        cfg = json.loads(core_config_path.read_text(encoding="utf-8"))
        rel_mesh = (cfg.get("solver") or {}).get("mesh_file")
        if rel_mesh:
            path = Path(str(rel_mesh))
            if not path.is_absolute():
                path = (repo_root / path).resolve()
            if path.is_file():
                return path
            raise FileNotFoundError(f"core_config solver.mesh_file missing: {path}")
    raise RuntimeError(
        "production_operator_mesh_unresolved: pass --operator-mesh-file or core_config.solver.mesh_file"
    )


def verify_operator_mesh_provenance(
    *,
    requested_mesh: Path,
    generated_mesh: Path,
    built: Mapping[str, Any],
    built_meta_path: Path,
    write_meta: bool = True,
) -> Dict[str, Any]:
    requested = requested_mesh.expanduser().resolve()
    generated = generated_mesh.expanduser().resolve()
    used_raw = built.get("operator_mesh_file_used")
    if not used_raw:
        raise RuntimeError("operator_mesh_file_used missing from operator build payload")
    used = Path(str(used_raw)).expanduser().resolve()
    if used != requested:
        raise RuntimeError(f"operator_mesh_provenance_mismatch: requested={requested} used={used}")
    if requested != generated:
        raise RuntimeError(
            f"operator_mesh_matches_generated=false: generated={generated} operator={requested}"
        )
    prov = {
        "generated_mesh_file": str(generated),
        "operator_mesh_file_used": str(used),
        "generated_mesh_sha256": _sha256_file(generated),
        "operator_mesh_sha256": _sha256_file(used),
        "operator_mesh_matches_generated": True,
        "dataset_version": DATASET_VERSION,
    }
    if write_meta and built_meta_path.is_file():
        meta = json.loads(built_meta_path.read_text(encoding="utf-8"))
        meta.update(prov)
        built_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return prov


def dolfinx_mesh_counts(mesh_path: Path) -> Dict[str, Optional[int]]:
    try:
        from v2_b3_synthesis_export import import_fem_main_3d  # noqa: WPS433

        fem3d, _ = import_fem_main_3d(start=Path(__file__).resolve().parent)  # type: ignore[attr-defined]
        msh, _, _ = fem3d._load_mesh_and_tags(mesh_path)
        tdim = msh.topology.dim
        return {
            "operator_node_count": int(msh.geometry.x.shape[0]),
            "operator_cell_count": int(msh.topology.index_map(tdim).size_local),
        }
    except Exception:
        return {"operator_node_count": None, "operator_cell_count": None}


def is_strict_production_mode(*, dataset_version: Optional[str] = None) -> bool:
    """Corrected-dataset production: strict fail-fast; env vars cannot disable."""
    ds = str(dataset_version or DATASET_VERSION).strip()
    return ds == DATASET_VERSION


def require_aperture_mask_production(*, dataset_version: Optional[str] = None) -> bool:
    if is_strict_production_mode(dataset_version=dataset_version):
        return True
    if os.environ.get(ALLOW_CAVITY_MAX_FALLBACK_ENV) == "1":
        return False
    if os.environ.get(DIAGNOSTIC_ONLY_FALLBACK_ENV) == "1":
        return False
    return os.environ.get(REQUIRE_APERTURE_MASK_ENV, "1") == "1"


def dataset_version_from_core_config(core_config_path: Optional[Path]) -> Optional[str]:
    if not core_config_path or not core_config_path.is_file():
        return None
    try:
        cfg = json.loads(core_config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    meta = cfg.get("m4_run_metadata") or {}
    ds = str(meta.get("dataset_version") or cfg.get("dataset_version") or "").strip()
    return ds or None


def aperture_export_required_for_core_config(core_config_path: Optional[Path]) -> bool:
    """True when corrected-dataset production or B3_REQUIRE_APERTURE_MASK mandates aperture export."""
    ds = dataset_version_from_core_config(core_config_path)
    if is_strict_production_mode(dataset_version=ds):
        return True
    if require_aperture_mask_production(dataset_version=ds):
        return True
    return ds == DATASET_VERSION


def resolve_production_region_dofs_mode(
    cli_value: Optional[str],
    *,
    core_config_path: Optional[Path],
    env_value: Optional[str] = None,
) -> str:
    """Production L_prod: corrected dataset always uses best_effort aperture export."""
    from v2_b3_checkpoint_pipeline_lib import resolve_synthesis_region_dofs_mode  # noqa: WPS433

    if aperture_export_required_for_core_config(core_config_path):
        return "best_effort"
    return resolve_synthesis_region_dofs_mode(cli_value, env_value=env_value)


def validate_pre_operator_build_region_dof_contract(
    *,
    region_dofs_mode: str,
    core_config_path: Optional[Path],
) -> list[str]:
    errors: list[str] = []
    if aperture_export_required_for_core_config(core_config_path) and region_dofs_mode != "best_effort":
        errors.append(
            f"region_dofs_mode={region_dofs_mode!r} forbidden for production "
            f"(dataset_version={DATASET_VERSION} or {REQUIRE_APERTURE_MASK_ENV}=1); "
            "use --B3-synthesis-region-dofs best_effort"
        )
    return errors


def validate_post_export_region_dof_contract(
    checkpoint_dir: Path,
    *,
    core_config_path: Optional[Path],
) -> list[str]:
    if not aperture_export_required_for_core_config(core_config_path):
        return []
    from v2_b3_rich_modal_lib import SYNTHESIS_METADATA_JSON, load_region_dof_bundle  # noqa: WPS433
    from v2_b3_synthesis_export import region_dof_status_is_pass  # noqa: WPS433

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    errors: list[str] = []
    synth_path = checkpoint_dir / SYNTHESIS_METADATA_JSON
    if synth_path.is_file():
        try:
            synth = json.loads(synth_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            synth = {}
        mode = str(synth.get("region_dof_indices_mode") or "")
        status = str(synth.get("region_dof_indices_status") or "")
        if mode == "off":
            errors.append(
                f"region_dof_indices_mode=off ({synth.get('region_dof_indices_error')})"
            )
        if status == "deferred_to_stage_c":
            errors.append(
                f"region_dof_indices_status=deferred_to_stage_c ({synth.get('region_dof_indices_error')})"
            )
        if not region_dof_status_is_pass(status):
            errors.append(f"region_dof_indices_status={status!r}")
    else:
        errors.append("missing synthesis_metadata.json")

    built_path = checkpoint_dir / "built_metadata.json"
    built_meta: Dict[str, Any] = {}
    if built_path.is_file():
        try:
            built_meta = json.loads(built_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            built_meta = {}
    try:
        load_region_dof_bundle(checkpoint_dir, built_meta, validate_aperture=True)
    except Exception as exc:
        errors.append(f"region_dof_bundle:{type(exc).__name__}:{exc}")
    return errors


def evaluate_production_acceptance(
    *,
    run_root: Path,
    sample_input: Mapping[str, Any],
) -> Dict[str, Any]:
    """Post-run acceptance contract for geometry-corrected production."""
    lprod = run_root / "lprod"
    ckpt = lprod / "checkpoint"
    built_path = ckpt / "built_metadata.json"
    agg_path = run_root / "aggregation" / "aggregation_result.json"
    catalog = run_root / "aggregation" / "modes_catalog.jsonl"
    rom_compare = run_root / "rom" / "rom_fom_compare_result.json"

    out: Dict[str, Any] = {
        "dataset_version": DATASET_VERSION,
        "acceptance_pass": False,
        "failures": [],
    }
    if not built_path.is_file():
        out["failures"].append("missing_built_metadata")
        return out

    built = json.loads(built_path.read_text(encoding="utf-8"))
    sample_id = str(sample_input.get("sample_id") or run_root.parent.parent.name)
    generated_mesh = lprod / "mesh" / "L_prod" / f"{sample_id}.msh"

    if not bool(built.get("operator_mesh_matches_generated")):
        out["failures"].append("operator_mesh_matches_generated!=true")
    if int(built.get("active_dimension") or 0) <= 0:
        out["failures"].append("active_dimension<=0")
    if not built.get("generated_mesh_sha256"):
        out["failures"].append("missing_generated_mesh_sha256")

    geom = extract_geometry_dict(sample_input)
    if not geom:
        geom = geometry_from_core_config(lprod / "resolved_core_config.json")
    fp = geometry_fingerprint(geom) if geom else None
    out["geometry_fingerprint"] = fp
    if not fp:
        out["failures"].append("geometry_fingerprint_missing")

    region_ctx = load_region_dof_bundle(ckpt, built)
    region = region_ctx.get("region") or {}
    p_raw = region.get("p_idx_aperture")
    p_ap = np.asarray([] if p_raw is None else p_raw, dtype=np.int32).ravel()
    out["p_idx_aperture_count"] = int(p_ap.size)
    if p_ap.size <= 0:
        out["failures"].append("p_idx_aperture_count<=0")

    if agg_path.is_file():
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        out["aggregation_status"] = agg.get("status")
        agg_status = str(agg.get("status") or "")
        if agg_status not in ("AGGREGATION_PASS", "PASS", "PASS_WITH_WARNINGS"):
            out["failures"].append(f"aggregation_status={agg.get('status')}")
        if not bool(agg.get("final_aggregation_ready")):
            out["failures"].append("final_aggregation_ready!=true")
    else:
        out["failures"].append("missing_aggregation_result")

    built_ds = str(built.get("dataset_version") or "")
    out["built_dataset_version"] = built_ds or None
    if built_ds and built_ds != DATASET_VERSION:
        out["failures"].append(f"dataset_version={built_ds}")
    elif not built_ds:
        out["failures"].append("missing_dataset_version")

    mic_methods = set()
    if catalog.is_file():
        import json as _json

        for line in catalog.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = _json.loads(line)
            except ValueError:
                continue
            m = row.get("mic_output_method")
            if m:
                mic_methods.add(str(m))
        out["catalog_mic_methods"] = sorted(mic_methods)
        if mic_methods and not mic_methods.issubset(ALLOWED_MIC_METHODS):
            out["failures"].append(f"mic_output_method_not_aperture_proxy:{sorted(mic_methods)}")
    else:
        out["failures"].append("missing_modes_catalog")

    out["rom_compare_present"] = rom_compare.is_file()
    out["mic_output_method"] = PRODUCTION_MIC_METHOD if not out["failures"] else None
    out["acceptance_pass"] = len(out["failures"]) == 0

    if is_strict_production_mode(dataset_version=DATASET_VERSION):
        out["failures"].extend(
            _evaluate_strict_production_failures(
                run_root=run_root,
                sample_id=sample_id,
                built=built,
                catalog=catalog,
                core_config_path=lprod / "resolved_core_config.json",
            )
        )
        out["strict_production_mode"] = True
        out["acceptance_pass"] = len(out["failures"]) == 0
    return out


def _evaluate_scout_strict_failures(run_root: Path) -> List[str]:
    """Fail-fast scout/target-plan gates for m4_geometry_corrected_v1 strict production."""
    failures: List[str] = []
    from v2_b3_run_coarse_scout_lhs_batch import _density_reference_coverage_ok  # noqa: WPS433

    density_path = run_root / "scout" / "discovery" / "density_result.json"
    if density_path.is_file():
        try:
            density = json.loads(density_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            failures.append("scout_density_result_unreadable")
            density = {}
        if density:
            ok, detail = _density_reference_coverage_ok(density)
            if not ok:
                failures.append(f"scout_density_intrinsic:{detail}")
            if str(density.get("status") or "") == "PARTIAL":
                failures.append("scout_density_status=PARTIAL")
            if density.get("coverage_policy") != "intrinsic_discovered_modes_v1":
                failures.append(f"scout_density_policy={density.get('coverage_policy') or 'missing'}")
    else:
        failures.append("missing_scout_density_result")

    scout_result_path = run_root / "scout" / "scout_result.json"
    if scout_result_path.is_file():
        try:
            scout_result = json.loads(scout_result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            failures.append("scout_result_unreadable")
            scout_result = {}
        discovery = scout_result.get("discovery") or {}
        exp_status = str(discovery.get("experiment_status") or scout_result.get("experiment_status") or "")
        if exp_status == "PARTIAL":
            failures.append("scout_discovery_experiment_status=PARTIAL")

    target_plan_path = run_root / "lprod" / "lprod_target_plan.json"
    if target_plan_path.is_file():
        try:
            target_plan = json.loads(target_plan_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            failures.append("lprod_target_plan_unreadable")
            target_plan = {}
        if bool(target_plan.get("placeholder")):
            failures.append("lprod_target_plan_placeholder=true")
        metadata = target_plan.get("target_metadata") or target_plan.get("targets_metadata") or []
        if isinstance(metadata, list):
            repair_count = sum(
                1
                for row in metadata
                if isinstance(row, dict) and str(row.get("source") or "") == "coverage_repair"
            )
            if repair_count and density_path.is_file():
                try:
                    density = json.loads(density_path.read_text(encoding="utf-8"))
                except (OSError, ValueError, json.JSONDecodeError):
                    density = {}
                if str(density.get("status") or "") != "PASS":
                    failures.append(f"lprod_target_plan_coverage_repair_from_non_pass_density:{repair_count}")

    for ckpt_rel in ("scout/checkpoint", "lprod/checkpoint"):
        synth_path = run_root / ckpt_rel / "synthesis_metadata.json"
        if not synth_path.is_file():
            continue
        try:
            synth = json.loads(synth_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            failures.append(f"{ckpt_rel.replace('/', '_')}_synthesis_metadata_unreadable")
            continue
        status = str(synth.get("region_dof_indices_status") or "")
        if status == "BEST_EFFORT_PASS":
            failures.append(f"{ckpt_rel.replace('/', '_')}_region_dof_indices_status=BEST_EFFORT_PASS")
        elif status and status not in ("present",):
            failures.append(f"{ckpt_rel.replace('/', '_')}_region_dof_indices_status={status}")

    return failures


def _evaluate_strict_production_failures(
    *,
    run_root: Path,
    sample_id: str,
    built: Mapping[str, Any],
    catalog: Path,
    core_config_path: Path,
) -> List[str]:
    """Additional fail-fast checks for m4_geometry_corrected_v1 (not overridable by env)."""
    failures: List[str] = list(_evaluate_scout_strict_failures(run_root))
    from v2_b3_m4_physics_identity_lib import (  # noqa: WPS433
        forbidden_solver_fallback_keys,
        scan_cross_sample_path_contamination,
    )

    contam = scan_cross_sample_path_contamination(run_root, sample_id=sample_id)
    if contam.get("contamination_detected"):
        hits = contam.get("contamination_hits") or []
        failures.append(f"cross_sample_path_reference:{hits[0].get('other_sample_id')}")

    op_used = str(built.get("operator_mesh_file_used") or "")
    if sample_id not in op_used.replace("\\", "/"):
        failures.append("operator_mesh_path_missing_sample_id")
    if "baseline_coupled_v2" in op_used and sample_id != "sample_000":
        failures.append("baseline_operator_mesh_selected")

    if core_config_path.is_file():
        try:
            cfg = json.loads(core_config_path.read_text(encoding="utf-8"))
            fb_keys = forbidden_solver_fallback_keys(cfg)
            if fb_keys:
                failures.append(f"forbidden_solver_fallback_config:{','.join(fb_keys)}")
        except (OSError, ValueError, json.JSONDecodeError):
            failures.append("resolved_core_config_unreadable")

    if catalog.is_file():
        import json as _json

        bad_participation = 0
        bad_mic = 0
        missing_lambda = 0
        missing_conv = 0
        solver_fb = 0
        for line in catalog.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = _json.loads(line)
            except ValueError:
                continue
            ps = str(row.get("participation_status") or "")
            if ps and ps != "computed":
                bad_participation += 1
            mic_m = str(row.get("mic_output_method") or "")
            if mic_m and mic_m != PRODUCTION_MIC_METHOD:
                bad_mic += 1
            if row.get("solver_fallback_used"):
                solver_fb += 1
            if row.get("frequency_hz") is not None and row.get("lambda_real") is None:
                missing_lambda += 1
            if (
                row.get("frequency_hz") is not None
                and row.get("eps_compute_error_relative") is None
                and row.get("convergence_status") != "converged"
            ):
                missing_conv += 1
        if bad_participation:
            failures.append(f"participation_status_not_computed:count={bad_participation}")
        if bad_mic:
            failures.append(f"mic_method_forbidden:count={bad_mic}")
        if solver_fb:
            failures.append(f"solver_fallback_used_in_catalog:count={solver_fb}")
        if missing_lambda:
            failures.append(f"missing_lambda_real_in_catalog:count={missing_lambda}")
        if missing_conv:
            failures.append(f"missing_convergence_evidence_in_catalog:count={missing_conv}")

    wr = run_root / "worker_results"
    if wr.is_dir():
        for chunk_dir in wr.iterdir():
            if not chunk_dir.is_dir():
                continue
            wr_path = chunk_dir / "worker_result.json"
            if wr_path.is_file():
                try:
                    wdoc = json.loads(wr_path.read_text(encoding="utf-8"))
                    if str(wdoc.get("status") or "") == "PARTIAL":
                        failures.append(f"worker_partial:{chunk_dir.name}")
                        break
                    if str(wdoc.get("status") or "") == "PASS_WITH_WARNING":
                        failures.append(f"worker_pass_with_warning:{chunk_dir.name}")
                        break
                except (OSError, ValueError, json.JSONDecodeError):
                    pass

    return failures
