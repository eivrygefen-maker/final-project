#!/usr/bin/env python3
"""Export aggregation plots/summaries to VM shared folder (Windows/OneDrive via VirtualBox)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_worker_run_lib import load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_SHARED_ROOT = Path("/media/sf_gmar")
SHARED_EXPORT_MANIFEST_SCHEMA = "m4_shared_export_manifest_v1"

PRIMARY_PLOT_NAMES = (
    "mode_frequency_vs_radiation_proxy.png",
    "mode_frequency_plot.png",
)

OPTIONAL_SUMMARY_FILES = (
    "modes_summary.json",
    "aggregation_result.json",
    "warnings_and_failures.json",
    "runtime_summary.json",
)


def detect_shared_root(explicit: Optional[Path] = None) -> Optional[Path]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        return path if path.is_dir() else None
    return DEFAULT_SHARED_ROOT if DEFAULT_SHARED_ROOT.is_dir() else None


def read_shape_name(run_root: Path) -> str:
    sample_input = run_root / "sample" / "sample_input.json"
    if sample_input.is_file():
        try:
            data = load_json(sample_input)
            shape = str(data.get("shape_name") or "").strip()
            if shape:
                return shape.lower()
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    agg = run_root / "aggregation" / "aggregation_result.json"
    if agg.is_file():
        try:
            data = load_json(agg)
            shape = str(data.get("shape_name") or "").strip()
            if shape:
                return shape.lower()
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return "classic"


def _export_filename(sample_id: str, run_id: str, artifact_name: str) -> str:
    stem = Path(artifact_name).stem
    suffix = Path(artifact_name).suffix
    return f"{sample_id}__{run_id}_{stem}{suffix}"


def export_sample_to_shared(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shared_root: Path,
    repo_root: Optional[Path] = None,
    shape_name: Optional[str] = None,
    copy_summaries: bool = True,
) -> Dict[str, Any]:
    """
    Copy main mode plot (+ optional summaries) to shared folder.
    Never raises — returns manifest with export_status EXPORTED | PARTIAL | SKIPPED | FAILED.
    """
    run_root = run_root.expanduser().resolve()
    shared_root = shared_root.expanduser().resolve()
    shape = (shape_name or read_shape_name(run_root)).strip().lower() or "classic"
    agg_dir = run_root / "aggregation"
    plots_dir = shared_root / shape / "plots"
    summaries_dir = shared_root / shape / "summaries"
    warnings: List[str] = []
    exported_files: List[str] = []
    shared_plot_path: Optional[str] = None
    source_plot: Optional[str] = None

    manifest: Dict[str, Any] = {
        "schema": SHARED_EXPORT_MANIFEST_SCHEMA,
        "sample_id": sample_id,
        "run_id": run_id,
        "shape_name": shape,
        "shared_root": str(shared_root),
        "source_plot": None,
        "shared_plot_path": None,
        "exported_files": [],
        "warnings": [],
        "export_status": "SKIPPED",
        "exported_at": utc_now(),
    }

    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        if copy_summaries:
            summaries_dir.mkdir(parents=True, exist_ok=True)

        plot_src: Optional[Path] = None
        for name in PRIMARY_PLOT_NAMES:
            candidate = agg_dir / name
            if candidate.is_file():
                plot_src = candidate
                source_plot = str(candidate)
                break
        if plot_src is None:
            warnings.append("no mode plot found in aggregation/")
            manifest["export_status"] = "FAILED"
            manifest["warnings"] = warnings
            _write_manifests(run_root, plots_dir, summaries_dir, manifest, repo_root)
            return manifest

        # Canonical shared name for Windows monitoring (content = radiation plot when available)
        dest_plot = plots_dir / _export_filename(sample_id, run_id, "mode_frequency_plot.png")
        shutil.copy2(plot_src, dest_plot)
        exported_files.append(str(dest_plot))
        shared_plot_path = str(dest_plot)
        manifest["source_plot"] = source_plot
        manifest["shared_plot_path"] = shared_plot_path
        if plot_src.name != "mode_frequency_plot.png":
            manifest["shared_plot_content"] = plot_src.name

        if copy_summaries:
            for fname in OPTIONAL_SUMMARY_FILES:
                src = agg_dir / fname
                if not src.is_file():
                    warnings.append(f"missing summary: {fname}")
                    continue
                dest = summaries_dir / _export_filename(sample_id, run_id, fname)
                shutil.copy2(src, dest)
                exported_files.append(str(dest))

        for extra_plot in (
            "mode_frequency_vs_mic_output_proxy.png",
            "mode_frequency_vs_bridge_excitation.png",
            "mode_frequency_vs_top_back_air_share.png",
        ):
            src = agg_dir / extra_plot
            if src.is_file():
                dest = plots_dir / _export_filename(sample_id, run_id, extra_plot)
                shutil.copy2(src, dest)
                exported_files.append(str(dest))

        manifest["exported_files"] = exported_files
        manifest["warnings"] = warnings
        manifest["export_status"] = "EXPORTED" if not warnings else "PARTIAL"
        _write_manifests(run_root, plots_dir, summaries_dir, manifest, repo_root)
        return manifest
    except OSError as exc:
        manifest["export_status"] = "FAILED"
        manifest["warnings"] = warnings + [str(exc)]
        try:
            _write_manifests(run_root, plots_dir, summaries_dir, manifest, repo_root)
        except OSError:
            pass
        return manifest


def _write_manifests(
    run_root: Path,
    plots_dir: Path,
    summaries_dir: Path,
    manifest: Mapping[str, Any],
    repo_root: Optional[Path],
) -> None:
    run_manifest = run_root / "aggregation" / "shared_export_manifest.json"
    write_json_atomic(run_manifest, dict(manifest))
    if manifest.get("sample_id") and manifest.get("run_id"):
        sid = str(manifest["sample_id"])
        rid = str(manifest["run_id"])
        shared_copy = summaries_dir / f"{sid}__{rid}_shared_export_manifest.json"
        try:
            summaries_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(shared_copy, dict(manifest))
        except OSError:
            pass
    if repo_root is not None:
        manifest_path = run_root / "aggregation" / "shared_export_manifest.json"
        if manifest_path.is_file():
            pass  # already written with rel paths optional


def try_export_sample_to_shared(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shared_root: Optional[Path],
    repo_root: Optional[Path] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Safe wrapper for production runner. Returns (manifest, warning_message)."""
    if shared_root is None:
        return None, "shared export skipped: shared root not found"
    if not shared_root.is_dir():
        return None, f"shared export skipped: not a directory: {shared_root}"
    manifest = export_sample_to_shared(
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        shared_root=shared_root,
        repo_root=repo_root,
    )
    if manifest.get("export_status") == "FAILED":
        return manifest, f"shared export failed: {manifest.get('warnings')}"
    if manifest.get("export_status") == "PARTIAL":
        return manifest, f"shared export partial: {manifest.get('warnings')}"
    return manifest, None
