#!/usr/bin/env python3
"""Export aggregation plots/summaries to VM shared folder (Windows/OneDrive via VirtualBox)."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_worker_run_lib import load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_SHARED_ROOT = Path("/media/sf_gmar")
M4_PRODUCTION_SUBFOLDER = "m4_production"
SHARED_EXPORT_MANIFEST_SCHEMA = "m4_shared_export_manifest_v1"
GRAPH_EXPORT_MANIFEST_SCHEMA = "m4_graph_export_manifest_v1"

PRIMARY_PLOT_NAMES = (
    "mode_frequency_vs_radiation_proxy.png",
    "mode_frequency_plot.png",
)

ALL_FINAL_PLOT_NAMES = (
    "mode_frequency_plot.png",
    "mode_frequency_vs_bridge_excitation.png",
    "mode_frequency_vs_mic_output_proxy.png",
    "mode_frequency_vs_radiation_proxy.png",
    "mode_frequency_vs_top_back_air_share.png",
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


def read_mesh_profile(run_root: Path) -> str:
    sample_input = run_root / "sample" / "sample_input.json"
    if sample_input.is_file():
        try:
            data = load_json(sample_input)
            profile = str(data.get("mesh_profile") or "").strip()
            if profile:
                return profile
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return "rom"


def graphs_destination_dir(
    *,
    shared_root: Path,
    shape_name: str,
    sample_id: str,
    run_id: str,
) -> Path:
    """Collision-safe graph package: {shared_root}/{shape}/m4_production/{sample_id}/{run_id}/graphs/"""
    shape = (shape_name or "classic").strip().lower() or "classic"
    return shared_root / shape / M4_PRODUCTION_SUBFOLDER / sample_id / run_id / "graphs"


def _export_filename(sample_id: str, run_id: str, artifact_name: str) -> str:
    stem = Path(artifact_name).stem
    suffix = Path(artifact_name).suffix
    return f"{sample_id}__{run_id}_{stem}{suffix}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_plot_with_proof(
    *,
    src: Path,
    dest: Path,
    plot_name: str,
) -> Dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    size = dest.stat().st_size
    row: Dict[str, Any] = {
        "plot_name": plot_name,
        "source_path": str(src),
        "destination_path": str(dest),
        "sha256": _sha256_file(dest),
        "bytes": size,
        "copy_status": "copied" if size > 0 else "empty_file",
    }
    if size <= 0:
        row["copy_status"] = "failed_empty"
    return row


def export_sample_to_shared(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shared_root: Path,
    repo_root: Optional[Path] = None,
    shape_name: Optional[str] = None,
    copy_summaries: bool = True,
    mesh_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Copy final aggregation plots (+ optional summaries) to shared folder.

    Writes collision-safe graph package under
    ``{shared_root}/{shape}/m4_production/{sample_id}/{run_id}/graphs/``
    and legacy flat ``{shape}/plots/`` names for monitoring compatibility.

    Never raises — returns manifest with export_status EXPORTED | PARTIAL | SKIPPED | FAILED.
    """
    run_root = run_root.expanduser().resolve()
    shared_root = shared_root.expanduser().resolve()
    shape = (shape_name or read_shape_name(run_root)).strip().lower() or "classic"
    profile = (mesh_profile or read_mesh_profile(run_root)).strip().lower() or "rom"
    agg_dir = run_root / "aggregation"
    plots_dir = shared_root / shape / "plots"
    summaries_dir = shared_root / shape / "summaries"
    graphs_dir = graphs_destination_dir(
        shared_root=shared_root,
        shape_name=shape,
        sample_id=sample_id,
        run_id=run_id,
    )
    warnings: List[str] = []
    exported_files: List[str] = []
    graph_entries: List[Dict[str, Any]] = []
    shared_plot_path: Optional[str] = None
    source_plot: Optional[str] = None

    manifest: Dict[str, Any] = {
        "schema": SHARED_EXPORT_MANIFEST_SCHEMA,
        "sample_id": sample_id,
        "run_id": run_id,
        "mesh_profile": profile,
        "shape_name": shape,
        "shared_root": str(shared_root),
        "graphs_dir": str(graphs_dir),
        "source_plot": None,
        "shared_plot_path": None,
        "exported_files": [],
        "graph_export_entries": [],
        "warnings": [],
        "export_status": "SKIPPED",
        "exported_at": utc_now(),
    }

    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)
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
            _write_manifests(run_root, graphs_dir, summaries_dir, manifest, repo_root)
            return manifest

        structured_primary = graphs_dir / plot_src.name
        entry = _copy_plot_with_proof(src=plot_src, dest=structured_primary, plot_name=plot_src.name)
        graph_entries.append(entry)
        if entry["copy_status"] != "copied":
            warnings.append(f"structured copy failed for {plot_src.name}")

        dest_plot = plots_dir / _export_filename(sample_id, run_id, "mode_frequency_plot.png")
        flat_entry = _copy_plot_with_proof(src=plot_src, dest=dest_plot, plot_name="mode_frequency_plot.png")
        graph_entries.append({**flat_entry, "destination_kind": "legacy_flat_plots"})
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

        for extra_plot in ALL_FINAL_PLOT_NAMES:
            if extra_plot == plot_src.name:
                continue
            src = agg_dir / extra_plot
            if not src.is_file():
                continue
            structured_dest = graphs_dir / extra_plot
            graph_entries.append(
                _copy_plot_with_proof(src=src, dest=structured_dest, plot_name=extra_plot)
            )
            flat_dest = plots_dir / _export_filename(sample_id, run_id, extra_plot)
            graph_entries.append(
                {
                    **_copy_plot_with_proof(src=src, dest=flat_dest, plot_name=extra_plot),
                    "destination_kind": "legacy_flat_plots",
                }
            )
            exported_files.append(str(flat_dest))
            exported_files.append(str(structured_dest))

        failed_copies = [e for e in graph_entries if e.get("copy_status") != "copied"]
        if failed_copies:
            warnings.append(f"graph_copy_failures={len(failed_copies)}")

        manifest["exported_files"] = sorted(set(exported_files))
        manifest["graph_export_entries"] = graph_entries
        manifest["warnings"] = warnings
        if failed_copies:
            manifest["export_status"] = "FAILED"
        elif warnings:
            manifest["export_status"] = "PARTIAL"
        else:
            manifest["export_status"] = "EXPORTED"
        _write_manifests(run_root, graphs_dir, summaries_dir, manifest, repo_root)
        return manifest
    except OSError as exc:
        manifest["export_status"] = "FAILED"
        manifest["warnings"] = warnings + [str(exc)]
        try:
            _write_manifests(run_root, graphs_dir, summaries_dir, manifest, repo_root)
        except OSError:
            pass
        return manifest


def _write_graph_export_manifest(graphs_dir: Path, manifest: Mapping[str, Any]) -> None:
    graph_manifest = {
        "schema": GRAPH_EXPORT_MANIFEST_SCHEMA,
        "generated_utc": utc_now(),
        "sample_id": manifest.get("sample_id"),
        "run_id": manifest.get("run_id"),
        "mesh_profile": manifest.get("mesh_profile"),
        "graphs_dir": str(graphs_dir),
        "entries": list(manifest.get("graph_export_entries") or []),
        "export_status": manifest.get("export_status"),
        "warnings": list(manifest.get("warnings") or []),
    }
    write_json_atomic(graphs_dir / "graph_export_manifest.json", graph_manifest)


def _write_manifests(
    run_root: Path,
    graphs_dir: Path,
    summaries_dir: Path,
    manifest: Mapping[str, Any],
    repo_root: Optional[Path],
) -> None:
    run_manifest = run_root / "aggregation" / "shared_export_manifest.json"
    write_json_atomic(run_manifest, dict(manifest))
    _write_graph_export_manifest(graphs_dir, manifest)
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
        _ = repo_root  # reserved for future rel-path stamping


def try_export_sample_to_shared(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shared_root: Optional[Path],
    repo_root: Optional[Path] = None,
    mesh_profile: Optional[str] = None,
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
        mesh_profile=mesh_profile,
    )
    if manifest.get("export_status") == "FAILED":
        return manifest, f"shared export failed: {manifest.get('warnings')}"
    if manifest.get("export_status") == "PARTIAL":
        return manifest, f"shared export partial: {manifest.get('warnings')}"
    return manifest, None


def export_graphs_fixture(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shared_root: Path,
    plot_names: Sequence[str] = ALL_FINAL_PLOT_NAMES,
) -> Dict[str, Any]:
    """Test helper: export from fixture aggregation dir without full pipeline."""
    agg = run_root / "aggregation"
    agg.mkdir(parents=True, exist_ok=True)
    for name in plot_names:
        path = agg / name
        if not path.is_file():
            path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    return export_sample_to_shared(
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        shared_root=shared_root,
        copy_summaries=False,
    )
