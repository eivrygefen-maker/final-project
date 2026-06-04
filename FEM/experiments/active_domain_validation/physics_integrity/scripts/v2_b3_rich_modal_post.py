#!/usr/bin/env python3
"""Stage C: region participation and audio output proxies from rich modal v1 export."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_checkpoint_metadata_lib import normalize_checkpoint_metadata  # noqa: E402
from v2_b3_checkpoint_pipeline_lib import (  # noqa: E402
    B3_SYNTHESIS_REGION_DOFS_ARG,
    B3_SYNTHESIS_REGION_DOFS_ENV,
    fail_with_messages,
    resolve_synthesis_region_dofs_mode,
    verify_rich_modal_post_environment,
)
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_rich_modal_lib import (  # noqa: E402
    MODES_ACTIVE_NPZ,
    REGION_DOF_INDICES_NPZ,
    RICH_MODAL_MANIFEST_JSON,
    RICH_MODAL_POST_SCHEMA,
    SYNTHESIS_METADATA_JSON,
    build_mode_synthesis_row,
    frequency_dedupe_report,
    load_region_dof_bundle,
    normalization_convention_v1,
)
STRUCTURAL_DEFERRED_WARNING = (
    "Structural region participation and displacement RMS proxies are unavailable "
    "(region_dof_indices.npz missing or deferred). Values are null, not physical zeros. "
    "Cavity pressure proxy may still be computed from active pressure DOFs (p_idx_air). "
    "Re-run Stage A with --B3-synthesis-region-dofs best_effort or Stage C with the same flag."
)


def _maybe_compute_region_dof_indices(
    checkpoint: Path,
    built_meta: Dict[str, Any],
    *,
    region_dofs_mode: str,
    synthesis_meta: Optional[Dict[str, Any]],
) -> List[str]:
    """Optional isolated subprocess region locate (Stage C). Returns warning strings."""
    warnings: List[str] = []
    if region_dofs_mode != "best_effort":
        return warnings
    if (checkpoint / REGION_DOF_INDICES_NPZ).is_file():
        return warnings

    mesh_level = str(
        (synthesis_meta or {}).get("mesh_level")
        or built_meta.get("mesh_level")
        or "L_prod"
    )
    from v2_b3_synthesis_export import export_region_dof_indices_isolated  # noqa: E402

    print(
        f"[B3_rich_modal_post] region_dof best_effort subprocess mesh_level={mesh_level}",
        flush=True,
    )
    status, error = export_region_dof_indices_isolated(
        checkpoint,
        mesh_level=mesh_level,
        built_meta=built_meta,
    )
    from v2_b3_synthesis_export import region_dof_status_is_pass  # noqa: E402

    if not region_dof_status_is_pass(status):
        warnings.append(
            f"Stage C region_dof best_effort did not produce npz: status={status} detail={error}"
        )
    else:
        print(f"[B3_rich_modal_post] region_dof_indices.npz written", flush=True)
    return warnings


def _collect_warnings(region_ctx: Dict[str, Any], extra: List[str]) -> List[str]:
    warnings = list(extra)
    if not region_ctx["structural_indices_available"]:
        warnings.append(STRUCTURAL_DEFERRED_WARNING)
    return warnings


def _write_post_md(path: Path, body: Dict[str, Any]) -> None:
    warnings = body.get("warnings") or []
    lines = [
        "# Rich modal post (Stage C)",
        "",
        f"- schema: `{body.get('schema')}`",
        f"- checkpoint_dir: `{body.get('checkpoint_dir')}`",
        f"- mode_count: `{body.get('mode_count')}`",
        f"- region_dof_source: `{body.get('region_dof_source')}`",
        f"- structural_region_participation_status: `{body.get('structural_region_participation_status')}`",
        f"- structural_audio_proxy_status: `{body.get('structural_audio_proxy_status')}`",
        f"- frequency_dedupe: `{body.get('frequency_dedupe')}`",
        "",
    ]
    if warnings:
        lines.extend(["## Warnings", ""])
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.extend(
        [
            "## Ranking by coverage (participation_top + participation_air_p; null = unavailable)",
            "",
            "| catalog_index | frequency_hz | st_shift_hz | participation_top | participation_air_p | top_rms_proxy |",
            "|---------------|--------------|-------------|-------------------|---------------------|---------------|",
        ]
    )
    for row in body.get("modes") or []:
        prox = row.get("audio_output_proxies") or {}
        lines.append(
            f"| {row.get('catalog_index')} | {row.get('frequency_hz')} | {row.get('st_shift_target_hz')} | "
            f"{row.get('participation_top')} | {row.get('participation_air_p')} | "
            f"{prox.get('top_plate_displacement_rms_proxy_v1')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_rich_modal_post(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stage C rich modal post (production .venv).")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--rich-modal-dir", required=True, help="Stage B rich_modal/ directory.")
    parser.add_argument("--output-dir", help="Default: parent of rich-modal-dir / rich_modal_post")
    parser.add_argument("--tolerance-hz", type=float, default=0.1)
    parser.add_argument(
        B3_SYNTHESIS_REGION_DOFS_ARG,
        dest="synthesis_region_dofs",
        choices=("off", "best_effort"),
        default="off",
        metavar="MODE",
        help=(
            "Optional Stage C region DOF locate in isolated subprocess (default off). "
            f"Env: {B3_SYNTHESIS_REGION_DOFS_ENV}=off|best_effort"
        ),
    )
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    try:
        region_dofs_mode = resolve_synthesis_region_dofs_mode(args.synthesis_region_dofs)
    except ValueError as exc:
        fail_with_messages("B3_rich_modal_post", [str(exc)])

    ok, messages = verify_rich_modal_post_environment(
        require_dolfinx=(region_dofs_mode == "best_effort"),
    )
    if not ok:
        fail_with_messages("B3_rich_modal_post", messages)

    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    rich_dir = Path(args.rich_modal_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else rich_dir.parent / "rich_modal_post"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = checkpoint / "built_metadata.json"
    synth_path = checkpoint / SYNTHESIS_METADATA_JSON
    rm_manifest_path = rich_dir / RICH_MODAL_MANIFEST_JSON
    modes_path = rich_dir / MODES_ACTIVE_NPZ
    if not meta_path.is_file() or not modes_path.is_file():
        fail_with_messages(
            "B3_rich_modal_post",
            [f"missing built_metadata or modes_active: {meta_path} / {modes_path}"],
        )

    synthesis_meta: Optional[Dict[str, Any]] = None
    if synth_path.is_file():
        synthesis_meta = json.loads(synth_path.read_text(encoding="utf-8"))

    built_meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
    built_meta, missing_keys, schema_pass = normalize_checkpoint_metadata(built_meta_raw)
    if not schema_pass:
        fail_with_messages(
            "B3_rich_modal_post",
            [f"built_metadata.json invalid or incomplete; missing keys: {missing_keys}"],
        )
    built = {
        "active_local": np.asarray(built_meta["active_local"], dtype=np.int32),
        "free_rows": np.asarray(built_meta["free_rows"], dtype=np.int32),
        "u_idx": np.asarray(built_meta["u_idx"], dtype=np.int32),
        "p_idx": np.asarray(built_meta["p_idx"], dtype=np.int32),
        "n_w": int(built_meta["n_w"]),
    }

    extra_warnings = _maybe_compute_region_dof_indices(
        checkpoint,
        built_meta,
        region_dofs_mode=region_dofs_mode,
        synthesis_meta=synthesis_meta,
    )
    region_ctx = load_region_dof_bundle(checkpoint, built_meta)
    warnings = _collect_warnings(region_ctx, extra_warnings)

    print(
        f"[B3_rich_modal_post] structural_indices_available="
        f"{region_ctx['structural_indices_available']} "
        f"pressure_indices_available={region_ctx['pressure_indices_available']} "
        f"region_dof_source={region_ctx['region_dof_source']}",
        flush=True,
    )
    if warnings:
        for w in warnings:
            print(f"[B3_rich_modal_post] WARN: {w}", flush=True)

    with np.load(modes_path, allow_pickle=False) as z:
        eig = np.asarray(z["eigenvectors_active"], dtype=np.float64)
        n_modes = int(eig.shape[1]) if eig.ndim == 2 else 0
        freq_hz = np.asarray(z["frequency_hz"], dtype=np.float64)
        lam_re = np.asarray(z["lambda_real"], dtype=np.float64)
        lam_im = np.asarray(z["lambda_imag"], dtype=np.float64)
        st_shift = np.asarray(z["st_shift_target_hz"], dtype=np.float64)
        target_idx = np.asarray(z["target_index"], dtype=np.int32)
        eps_slot = np.asarray(z["eps_slot_index"], dtype=np.int32)
        eps_err = np.asarray(z["eps_compute_error_relative"], dtype=np.float64)
        u_norm = np.asarray(z["u_norm_W"], dtype=np.float64)
        p_norm = np.asarray(z["p_norm_W"], dtype=np.float64)
        p_sup = np.asarray(z["p_support"], dtype=np.float64)

    modes_out: List[Dict[str, Any]] = []
    for j in range(n_modes):
        modes_out.append(
            build_mode_synthesis_row(
                catalog_index=int(j),
                x_active=eig[:, j],
                built=built,
                region_ctx=region_ctx,
                scalars={
                    "frequency_hz": float(freq_hz[j]),
                    "lambda_real": float(lam_re[j]),
                    "lambda_imag": float(lam_im[j]),
                    "st_shift_target_hz": float(st_shift[j]),
                    "target_index": int(target_idx[j]),
                    "eps_slot_index": int(eps_slot[j]),
                    "eps_compute_error_relative": float(eps_err[j]),
                    "u_norm_W": float(u_norm[j]),
                    "p_norm_W": float(p_norm[j]),
                    "p_support": float(p_sup[j]),
                },
            )
        )

    dedupe = frequency_dedupe_report(modes_out, tol_hz=float(args.tolerance_hz))
    body: Dict[str, Any] = {
        "schema": RICH_MODAL_POST_SCHEMA,
        "checkpoint_dir": str(checkpoint),
        "rich_modal_dir": str(rich_dir),
        "output_dir": str(output_dir),
        "synthesis_metadata": synthesis_meta,
        "rich_modal_manifest": json.loads(rm_manifest_path.read_text(encoding="utf-8"))
        if rm_manifest_path.is_file()
        else None,
        "normalization_convention": normalization_convention_v1(),
        "mode_count": len(modes_out),
        "modes": modes_out,
        "frequency_dedupe": dedupe,
        "region_dof_source": region_ctx["region_dof_source"],
        "region_dof_indices_npz": str((checkpoint / REGION_DOF_INDICES_NPZ).resolve())
        if region_ctx["npz_present"]
        else None,
        "region_dof_indices_status": (
            synthesis_meta.get("region_dof_indices_status") if synthesis_meta else None
        ),
        "structural_region_participation_status": region_ctx["structural_region_participation_status"],
        "structural_audio_proxy_status": region_ctx["structural_audio_proxy_status"],
        "pressure_region_status": region_ctx["pressure_region_status"],
        "stage_c_region_dofs_mode": region_dofs_mode,
        "warnings": warnings,
    }
    write_json_atomic(output_dir / "modes_synthesis.json", body)
    write_json_atomic(
        output_dir / "rich_modal_post_manifest.json",
        {
            "schema": RICH_MODAL_POST_SCHEMA,
            "modes_synthesis_json": str((output_dir / "modes_synthesis.json").resolve()),
            "mode_count": len(modes_out),
            "frequency_dedupe": dedupe,
            "structural_region_participation_status": body["structural_region_participation_status"],
            "structural_audio_proxy_status": body["structural_audio_proxy_status"],
            "region_dof_source": body["region_dof_source"],
            "warnings": warnings,
        },
    )
    _write_post_md(output_dir / "modes_synthesis.md", body)
    print(
        f"[B3_rich_modal_post] modes={len(modes_out)} duplicate_groups={dedupe.get('duplicate_groups')} "
        f"structural_status={body['structural_region_participation_status']} "
        f"-> {output_dir / 'modes_synthesis.json'}",
        flush=True,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    return run_rich_modal_post(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
