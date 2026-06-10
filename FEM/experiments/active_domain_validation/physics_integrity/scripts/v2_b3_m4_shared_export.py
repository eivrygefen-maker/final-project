#!/usr/bin/env python3
"""Export approved aggregation plots and compact summaries to the VM shared folder."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from v2_b3_m4_lhs_pool_bridge import is_run_usably_complete, read_run_production_summary  # noqa: E402
from v2_b3_m4_worker_run_lib import load_json, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

DEFAULT_SHARED_ROOT = Path("/media/sf_gmar")
SHARED_EXPORT_MANIFEST_SCHEMA = "m4_shared_export_manifest_v2"
GRAPH_EXPORT_MANIFEST_SCHEMA = "m4_graph_export_manifest_v1"
SUMMARY_SCHEMA = "m4_shared_summary_v1"

APPROVED_SHARED_PLOT_NAMES: Tuple[str, ...] = (
    "mode_frequency_vs_bridge_excitation.png",
    "mode_frequency_vs_mic_output_proxy.png",
    "mode_frequency_vs_radiation_proxy.png",
    "mode_frequency_vs_top_back_air_share.png",
)

EXCLUDED_SHARED_PLOT_NAMES: Tuple[str, ...] = (
    "mode_frequency_plot.png",
)

LEGACY_NESTED_SUBFOLDER = "m4_production"
LEGACY_GRAPHS_DIRNAME = "graphs"


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


def sample_plots_destination_dir(
    *,
    shared_root: Path,
    shape_name: str,
    sample_id: str,
) -> Path:
    """``{shared_root}/{shape}/plots/{sample_id}/``"""
    shape = (shape_name or "classic").strip().lower() or "classic"
    return shared_root / shape / "plots" / sample_id


def summaries_destination_dir(*, shared_root: Path, shape_name: str) -> Path:
    """``{shared_root}/{shape}/summaries/``"""
    shape = (shape_name or "classic").strip().lower() or "classic"
    return shared_root / shape / "summaries"


def run_plot_filename(run_id: str, plot_name: str) -> str:
    return f"{run_id}__{plot_name}"


def summary_json_filename(sample_id: str, run_id: str) -> str:
    return f"{sample_id}__{run_id}__summary.json"


def graph_manifest_filename(sample_id: str, run_id: str) -> str:
    return f"{sample_id}__{run_id}__graph_export_manifest.json"


def approved_plots_present_in_aggregation(run_root: Path) -> List[str]:
    agg_dir = run_root / "aggregation"
    return [name for name in APPROVED_SHARED_PLOT_NAMES if (agg_dir / name).is_file()]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha(repo_root: Optional[Path]) -> Optional[str]:
    if repo_root is None:
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _safe_shared_replace(tmp: Path, dest: Path) -> None:
    """File-level replace on shared mounts; never directory replace."""
    try:
        os.replace(tmp, dest)
    except OSError:
        shutil.copy2(tmp, dest)
        try:
            tmp.unlink()
        except OSError:
            pass


def _safe_copy_file(*, src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(src, tmp)
        _safe_shared_replace(tmp, dest)
    except Exception:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        shutil.copy2(src, dest)


def _safe_write_json_shared(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        _safe_shared_replace(tmp, path)
    except Exception:
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        path.write_text(text, encoding="utf-8")


def _best_effort_unlink_file(path: Path) -> Tuple[bool, Optional[str], bool]:
    if not path.is_file():
        return False, None, False
    try:
        path.unlink()
        return True, None, False
    except OSError as exc:
        permission = exc.errno in (1, 13) or "not permitted" in str(exc).lower() or "permission" in str(exc).lower()
        return False, str(exc), permission


def _legacy_stale_file_candidates(
    *,
    shared_root: Path,
    shape: str,
    sample_id: str,
    run_id: str,
) -> List[Path]:
    legacy_names = (
        f"{sample_id}__{run_id}_mode_frequency_plot.png",
        f"{sample_id}__{run_id}__mode_frequency_plot.png",
        run_plot_filename(run_id, "mode_frequency_plot.png"),
    )
    candidates: List[Path] = []
    plots_root = shared_root / shape / "plots"
    sample_plots = sample_plots_destination_dir(
        shared_root=shared_root,
        shape_name=shape,
        sample_id=sample_id,
    )
    for name in legacy_names:
        candidates.append(plots_root / name)
        candidates.append(sample_plots / name)

    run_legacy_root = shared_root / shape / LEGACY_NESTED_SUBFOLDER / sample_id / run_id
    graphs_dir = run_legacy_root / LEGACY_GRAPHS_DIRNAME
    if graphs_dir.is_dir():
        for child in graphs_dir.iterdir():
            if child.is_file():
                candidates.append(child)
    if run_legacy_root.is_dir():
        for child in run_legacy_root.iterdir():
            if child.is_file():
                candidates.append(child)

    unique: Dict[str, Path] = {}
    for path in candidates:
        unique[str(path)] = path
    return list(unique.values())


def remove_stale_shared_exports_for_run(
    *,
    shared_root: Path,
    shape_name: str,
    sample_id: str,
    run_id: str,
) -> Dict[str, Any]:
    """
    Best-effort removal of exact stale files for this sample/run only.

    Never deletes, chmods, renames, or recursively removes shared directories.
  """
    shared_root = shared_root.expanduser().resolve()
    shape = (shape_name or "classic").strip().lower() or "classic"
    removed_files: List[str] = []
    errors: List[str] = []
    permission_errors = 0

    for path in _legacy_stale_file_candidates(
        shared_root=shared_root,
        shape=shape,
        sample_id=sample_id,
        run_id=run_id,
    ):
        removed, err, permission = _best_effort_unlink_file(path)
        if removed:
            removed_files.append(str(path))
        elif err:
            errors.append(f"{path}: {err}")
            if permission:
                permission_errors += 1

    if not errors:
        status = "COMPLETED"
    elif permission_errors:
        status = "SKIPPED_PERMISSION"
    else:
        status = "PARTIAL"

    return {
        "legacy_cleanup_status": status,
        "legacy_cleanup_errors": errors,
        "removed_stale_files": removed_files,
    }


def _copy_plot_with_proof(*, src: Path, dest: Path, plot_name: str) -> Dict[str, Any]:
    _safe_copy_file(src=src, dest=dest)
    size = dest.stat().st_size
    digest = _sha256_file(dest)
    row: Dict[str, Any] = {
        "plot_name": plot_name,
        "source_path": str(src),
        "destination_path": str(dest),
        "sha256": digest,
        "size_bytes": size,
        "copy_status": "copied" if size > 0 else "failed_empty",
    }
    return row


def build_compact_summary_payload(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    export_manifest: Mapping[str, Any],
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    sample_input = load_json(run_root / "sample" / "sample_input.json") if (
        run_root / "sample" / "sample_input.json"
    ).is_file() else {}
    summary = read_run_production_summary(run_root)
    prov_path = run_root / "m4_sample_runtime_provenance.json"
    prov = load_json(prov_path) if prov_path.is_file() else {}
    runtime = load_json(run_root / "aggregation" / "runtime_summary.json") if (
        run_root / "aggregation" / "runtime_summary.json"
    ).is_file() else {}
    stage_times = prov.get("stage_wall_times_s") or runtime.get("stage_wall_times_s") or {}
    barrier = load_json(run_root / "cleanup" / "sample_cleanup_barrier.json") if (
        run_root / "cleanup" / "sample_cleanup_barrier.json"
    ).is_file() else {}
    compaction = barrier.get("compaction") if isinstance(barrier.get("compaction"), dict) else {}
    graph_entries = list(export_manifest.get("graph_export_entries") or [])
    graph_destinations = [e.get("destination_path") for e in graph_entries if e.get("destination_path")]

    return {
        "schema": SUMMARY_SCHEMA,
        "generated_utc": utc_now(),
        "sample_id": sample_id,
        "run_id": run_id,
        "lhs_index": sample_input.get("lhs_row_index"),
        "mesh_profile": sample_input.get("mesh_profile") or export_manifest.get("mesh_profile"),
        "mesh_level_id": sample_input.get("mesh_level_id"),
        "dataset_version": sample_input.get("dataset_version"),
        "terminal_status": summary.get("terminal_status"),
        "aggregation_status": summary.get("aggregation_status"),
        "raw_mode_count": summary.get("raw_modes"),
        "deduped_mode_count": summary.get("deduped_modes"),
        "completed_chunks": summary.get("completed_chunks"),
        "planned_chunks": summary.get("planned_chunks"),
        "workers_actual_parallel": summary.get("workers_actual_parallel"),
        "total_runtime_s": prov.get("total_runtime_s") or runtime.get("total_runtime_s"),
        "scout_runtime_s": stage_times.get("stage1_scout_mesh")
        or stage_times.get("stage2_scout_discovery"),
        "checkpoint_runtime_s": stage_times.get("stage4_lprod_export"),
        "worker_runtime_s": stage_times.get("stage5_workers"),
        "freeze_runtime_s": stage_times.get("stage6_freeze"),
        "peak_rss_bytes_per_worker": prov.get("peak_rss_bytes_per_worker")
        or runtime.get("peak_rss_bytes_per_worker"),
        "cleanup_status": barrier.get("status"),
        "compaction_status": compaction.get("status"),
        "graph_export_status": export_manifest.get("export_status"),
        "graph_destination_paths": graph_destinations,
        "legacy_cleanup_status": export_manifest.get("legacy_cleanup_status"),
        "git_commit_sha": _git_head_sha(repo_root),
    }


def export_sample_to_shared(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shared_root: Path,
    repo_root: Optional[Path] = None,
    shape_name: Optional[str] = None,
    mesh_profile: Optional[str] = None,
    cleanup_stale_exports: bool = True,
) -> Dict[str, Any]:
    """
    Export approved PNG plots to ``{shared_root}/{shape}/plots/{sample_id}/`` and compact
    summary + graph manifest to ``{shared_root}/{shape}/summaries/``.

    Never raises — returns manifest with export_status EXPORTED | FAILED.
    Legacy cleanup failures never block successful new exports.
    """
    run_root = run_root.expanduser().resolve()
    shared_root = shared_root.expanduser().resolve()
    shape = (shape_name or read_shape_name(run_root)).strip().lower() or "classic"
    profile = (mesh_profile or read_mesh_profile(run_root)).strip().lower() or "rom"
    agg_dir = run_root / "aggregation"
    plots_dir = sample_plots_destination_dir(
        shared_root=shared_root,
        shape_name=shape,
        sample_id=sample_id,
    )
    summaries_dir = summaries_destination_dir(shared_root=shared_root, shape_name=shape)
    warnings: List[str] = []
    graph_entries: List[Dict[str, Any]] = []
    exported_png_paths: List[str] = []

    manifest: Dict[str, Any] = {
        "schema": SHARED_EXPORT_MANIFEST_SCHEMA,
        "sample_id": sample_id,
        "run_id": run_id,
        "mesh_profile": profile,
        "shape_name": shape,
        "shared_root": str(shared_root),
        "plots_dir": str(plots_dir),
        "summaries_dir": str(summaries_dir),
        "approved_plot_names": list(APPROVED_SHARED_PLOT_NAMES),
        "excluded_plot_names": list(EXCLUDED_SHARED_PLOT_NAMES),
        "graph_export_entries": [],
        "summary_export_path": None,
        "graph_manifest_export_path": None,
        "legacy_cleanup_status": "SKIPPED",
        "legacy_cleanup_errors": [],
        "warnings": [],
        "export_status": "SKIPPED",
        "exported_at": utc_now(),
    }

    if cleanup_stale_exports:
        cleanup = remove_stale_shared_exports_for_run(
            shared_root=shared_root,
            shape_name=shape,
            sample_id=sample_id,
            run_id=run_id,
        )
        manifest["legacy_cleanup_status"] = cleanup.get("legacy_cleanup_status")
        manifest["legacy_cleanup_errors"] = list(cleanup.get("legacy_cleanup_errors") or [])
        if cleanup.get("removed_stale_files"):
            manifest["removed_stale_files"] = list(cleanup["removed_stale_files"])

    try:
        plots_dir.mkdir(parents=True, exist_ok=True)
        summaries_dir.mkdir(parents=True, exist_ok=True)

        present_in_agg = approved_plots_present_in_aggregation(run_root)
        if not present_in_agg:
            warnings.append("no approved plots found in aggregation/")
            manifest["export_status"] = "FAILED"
            manifest["warnings"] = warnings
            _write_run_local_manifest(run_root, manifest)
            return manifest

        for plot_name in present_in_agg:
            src = agg_dir / plot_name
            dest = plots_dir / run_plot_filename(run_id, plot_name)
            entry = _copy_plot_with_proof(src=src, dest=dest, plot_name=plot_name)
            graph_entries.append(entry)
            if entry["copy_status"] == "copied":
                exported_png_paths.append(str(dest))

        failed_copies = [e for e in graph_entries if e.get("copy_status") != "copied"]
        if failed_copies:
            warnings.append(f"graph_copy_failures={len(failed_copies)}")
            manifest["export_status"] = "FAILED"
            manifest["graph_export_entries"] = graph_entries
            manifest["warnings"] = warnings
            _write_run_local_manifest(run_root, manifest)
            return manifest

        manifest["graph_export_entries"] = graph_entries
        manifest["exported_png_paths"] = exported_png_paths

        summary_payload = build_compact_summary_payload(
            run_root=run_root,
            sample_id=sample_id,
            run_id=run_id,
            export_manifest=manifest,
            repo_root=repo_root,
        )
        summary_path = summaries_dir / summary_json_filename(sample_id, run_id)
        _safe_write_json_shared(summary_path, summary_payload)
        manifest["summary_export_path"] = str(summary_path)

        graph_manifest = {
            "schema": GRAPH_EXPORT_MANIFEST_SCHEMA,
            "generated_utc": utc_now(),
            "sample_id": sample_id,
            "run_id": run_id,
            "mesh_profile": profile,
            "plots_dir": str(plots_dir),
            "summaries_dir": str(summaries_dir),
            "approved_plot_names": list(present_in_agg),
            "entries": graph_entries,
            "export_status": "EXPORTED",
            "warnings": warnings,
            "legacy_cleanup_status": manifest.get("legacy_cleanup_status"),
            "legacy_cleanup_errors": manifest.get("legacy_cleanup_errors"),
        }
        graph_manifest_path = summaries_dir / graph_manifest_filename(sample_id, run_id)
        _safe_write_json_shared(graph_manifest_path, graph_manifest)
        manifest["graph_manifest_export_path"] = str(graph_manifest_path)

        manifest["warnings"] = warnings
        manifest["export_status"] = "EXPORTED"
        _write_run_local_manifest(run_root, manifest)
        return manifest
    except OSError as exc:
        manifest["export_status"] = "FAILED"
        manifest["warnings"] = warnings + [str(exc)]
        try:
            _write_run_local_manifest(run_root, manifest)
        except OSError:
            pass
        return manifest


def _write_run_local_manifest(run_root: Path, manifest: Mapping[str, Any]) -> None:
    write_json_atomic(run_root / "aggregation" / "shared_export_manifest.json", dict(manifest))


def try_export_sample_to_shared(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shared_root: Optional[Path],
    repo_root: Optional[Path] = None,
    mesh_profile: Optional[str] = None,
    cleanup_stale_exports: bool = True,
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
        cleanup_stale_exports=cleanup_stale_exports,
    )
    if manifest.get("export_status") == "FAILED":
        return manifest, f"shared export failed: {manifest.get('warnings')}"
    return manifest, None


def export_graphs_fixture(
    *,
    run_root: Path,
    sample_id: str,
    run_id: str,
    shared_root: Path,
    plot_names: Sequence[str] = APPROVED_SHARED_PLOT_NAMES,
) -> Dict[str, Any]:
    """Test helper: export from fixture aggregation dir without full pipeline."""
    agg = run_root / "aggregation"
    agg.mkdir(parents=True, exist_ok=True)
    for name in plot_names:
        path = agg / name
        if not path.is_file():
            path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    if (agg / "mode_frequency_plot.png").is_file() is False and "mode_frequency_plot.png" not in plot_names:
        (agg / "mode_frequency_plot.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    return export_sample_to_shared(
        run_root=run_root,
        sample_id=sample_id,
        run_id=run_id,
        shared_root=shared_root,
    )


def verify_numerical_success(run_root: Path) -> Tuple[bool, Dict[str, Any]]:
    summary = read_run_production_summary(run_root)
    return is_run_usably_complete(summary) and str(summary.get("terminal_status") or "") == "COMPLETED", summary
