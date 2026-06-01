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

from v2_b3_checkpoint_pipeline_lib import verify_production_stage_environment, fail_with_messages  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402
from v2_b3_rich_modal_lib import (  # noqa: E402
    MODES_ACTIVE_NPZ,
    MODES_CATALOG_JSONL,
    REGION_DOF_INDICES_NPZ,
    RICH_MODAL_DIRNAME,
    RICH_MODAL_MANIFEST_JSON,
    RICH_MODAL_POST_SCHEMA,
    SYNTHESIS_METADATA_JSON,
    frequency_dedupe_report,
    normalization_convention_v1,
    prolongate_active_to_W,
    participation_energy_fraction,
)
from v2_b3_st_sinvert_solver_lib import normalize_checkpoint_metadata  # noqa: E402


def _load_region_indices(checkpoint: Path, built_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    path = checkpoint / REGION_DOF_INDICES_NPZ
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            return {k: np.asarray(z[k]).ravel() for k in z.files if k != "layout"}
    u_idx = np.asarray(built_meta["u_idx"], dtype=np.int32).ravel()
    p_idx = np.asarray(built_meta["p_idx"], dtype=np.int32).ravel()
    return {
        "u_idx_top": np.asarray([], dtype=np.int32),
        "u_idx_back": np.asarray([], dtype=np.int32),
        "u_idx_ribs": np.asarray([], dtype=np.int32),
        "u_idx_soundhole": np.asarray([], dtype=np.int32),
        "p_idx_air": p_idx.copy(),
        "p_idx_all": p_idx.copy(),
        "u_idx_all": u_idx.copy(),
    }


def _audio_output_proxies(
    x_full: np.ndarray,
    region: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    u_top = region.get("u_idx_top", np.asarray([], dtype=np.int32))
    u_sh = region.get("u_idx_soundhole", np.asarray([], dtype=np.int32))
    p_air = region.get("p_idx_air", np.asarray([], dtype=np.int32))

    top_vals = x_full[np.asarray(u_top, dtype=np.int32)] if u_top.size else np.asarray([])
    sh_vals = x_full[np.asarray(u_sh, dtype=np.int32)] if u_sh.size else np.asarray([])
    p_vals = x_full[np.asarray(p_air, dtype=np.int32)] if p_air.size else np.asarray([])

    return {
        "top_plate_displacement_rms_proxy_v1": float(np.sqrt(np.mean(top_vals**2))) if top_vals.size else 0.0,
        "soundhole_facet_displacement_rms_proxy_v1": float(np.sqrt(np.mean(sh_vals**2))) if sh_vals.size else 0.0,
        "cavity_pressure_max_proxy_v1": float(np.max(np.abs(p_vals))) if p_vals.size else 0.0,
        "proxy_type": "audio_output_proxy_v1",
        "not_microphone_pressure": True,
    }


def _write_post_md(path: Path, body: Dict[str, Any]) -> None:
    lines = [
        "# Rich modal post (Stage C)",
        "",
        f"- schema: `{body.get('schema')}`",
        f"- checkpoint_dir: `{body.get('checkpoint_dir')}`",
        f"- mode_count: `{body.get('mode_count')}`",
        f"- frequency_dedupe: `{body.get('frequency_dedupe')}`",
        "",
        "## Ranking by coverage (participation_top + participation_air_p)",
        "",
        "| catalog_index | frequency_hz | st_shift_hz | participation_top | participation_air_p | top_rms_proxy |",
        "|---------------|--------------|-------------|-------------------|---------------------|---------------|",
    ]
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
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    ok, messages = verify_production_stage_environment()
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

    built_meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
    built_meta, _missing, _schema_pass = normalize_checkpoint_metadata(built_meta_raw)
    built = {
        "active_local": np.asarray(built_meta["active_local"], dtype=np.int32),
        "free_rows": np.asarray(built_meta["free_rows"], dtype=np.int32),
        "u_idx": np.asarray(built_meta["u_idx"], dtype=np.int32),
        "p_idx": np.asarray(built_meta["p_idx"], dtype=np.int32),
        "n_w": int(built_meta["n_w"]),
    }
    region = _load_region_indices(checkpoint, built_meta)

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
        x_active = eig[:, j]
        x_full = prolongate_active_to_W(x_active, built)
        row: Dict[str, Any] = {
            "catalog_index": int(j),
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
            "participation_top": participation_energy_fraction(x_full, region.get("u_idx_top", [])),
            "participation_back": participation_energy_fraction(x_full, region.get("u_idx_back", [])),
            "participation_ribs": participation_energy_fraction(x_full, region.get("u_idx_ribs", [])),
            "participation_air_p": participation_energy_fraction(x_full, region.get("p_idx_air", [])),
            "audio_output_proxies": _audio_output_proxies(x_full, region),
        }
        modes_out.append(row)

    dedupe = frequency_dedupe_report(modes_out, tol_hz=float(args.tolerance_hz))
    body: Dict[str, Any] = {
        "schema": RICH_MODAL_POST_SCHEMA,
        "checkpoint_dir": str(checkpoint),
        "rich_modal_dir": str(rich_dir),
        "output_dir": str(output_dir),
        "synthesis_metadata": json.loads(synth_path.read_text(encoding="utf-8")) if synth_path.is_file() else None,
        "rich_modal_manifest": json.loads(rm_manifest_path.read_text(encoding="utf-8"))
        if rm_manifest_path.is_file()
        else None,
        "normalization_convention": normalization_convention_v1(),
        "mode_count": len(modes_out),
        "modes": modes_out,
        "frequency_dedupe": dedupe,
        "region_dof_source": str(checkpoint / REGION_DOF_INDICES_NPZ)
        if (checkpoint / REGION_DOF_INDICES_NPZ).is_file()
        else "built_metadata_fallback",
    }
    write_json_atomic(output_dir / "modes_synthesis.json", body)
    write_json_atomic(
        output_dir / "rich_modal_post_manifest.json",
        {
            "schema": RICH_MODAL_POST_SCHEMA,
            "modes_synthesis_json": str((output_dir / "modes_synthesis.json").resolve()),
            "mode_count": len(modes_out),
            "frequency_dedupe": dedupe,
        },
    )
    _write_post_md(output_dir / "modes_synthesis.md", body)
    print(
        f"[B3_rich_modal_post] modes={len(modes_out)} duplicate_groups={dedupe.get('duplicate_groups')} "
        f"-> {output_dir / 'modes_synthesis.json'}",
        flush=True,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    return run_rich_modal_post(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
