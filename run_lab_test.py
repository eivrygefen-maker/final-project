#!/usr/bin/env python3
"""
Isolated FEM lab run: copy config + mesh into LAB/test_run_<timestamp>/, sweep with
``fem_master_dynamic`` into a private SORTING tree (no global candidates_log / ROM),
then MMR + package ROM under the lab folder only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple


def _repo_root() -> Path:
    repo = Path(os.path.abspath(__file__)).resolve()
    while repo.name != "final-project" and repo.parent != repo:
        repo = repo.parent
    if repo.name != "final-project":
        raise RuntimeError(
            "Could not locate parent directory named 'final-project' starting from "
            f"{Path(__file__).resolve()}"
        )
    return repo


def _resolve_mesh_source(cfg: Dict[str, Any], config_path: Path, repo: Path) -> Path:
    raw = Path(str(cfg.get("solver", {}).get("mesh_file", "")))
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, config_path.parents[1], repo / "FEM", repo):
        cand = (base / raw).resolve()
        if cand.is_file():
            return cand
    return (repo / raw).resolve()


def _prepare_lab(
    repo: Path,
    lab: Path,
    source_config: Path,
) -> Tuple[Path, Path]:
    """Copy guitar_3d.json + mesh into lab; return (lab_config_path, lab_mesh_path)."""
    lab_cfg_dir = lab / "FEM" / "configs"
    lab_mesh_dir = lab / "FEM" / "mesh"
    lab_sort = lab / "SORTING"
    lab_rom = lab / "rom"
    for d in (lab_cfg_dir, lab_mesh_dir, lab_sort / "temp_modes", lab_sort / "temp_results", lab_rom):
        d.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(source_config.read_text(encoding="utf-8"))
    mesh_src = _resolve_mesh_source(cfg, source_config.resolve(), repo)
    if not mesh_src.is_file():
        raise FileNotFoundError(f"Mesh not found for lab copy: {mesh_src}")

    lab_mesh = lab_mesh_dir / mesh_src.name
    shutil.copy2(mesh_src, lab_mesh)
    # Optional sidecar (e.g. some workflows keep .h5 next to mesh name).
    for suffix in (".h5", ".xdmf"):
        side = mesh_src.with_suffix(suffix)
        if side.is_file():
            shutil.copy2(side, lab_mesh.with_suffix(suffix))

    cfg.setdefault("solver", {})
    cfg["solver"]["mesh_file"] = str(lab_mesh.resolve())

    lab_config = lab_cfg_dir / "guitar_3d.json"
    lab_config.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    (lab_sort / "candidates_log.json").write_text(
        json.dumps({"candidates": []}, indent=2) + "\n", encoding="utf-8"
    )

    meta = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source_config": str(source_config.resolve()),
        "source_mesh": str(mesh_src.resolve()),
        "lab_config": str(lab_config.resolve()),
        "lab_mesh": str(lab_mesh.resolve()),
    }
    (lab / "lab_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return lab_config, lab_mesh


def _run_step(name: str, cmd: list[str], cwd: Path) -> int:
    print(f"\n{'=' * 72}\n  {name}\n  $ {' '.join(cmd)}\n{'=' * 72}")
    sys.stdout.flush()
    return int(subprocess.run(cmd, cwd=str(cwd)).returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sandbox FEM sweep + ROM packaging under LAB/test_run_<timestamp>/ only."
    )
    parser.add_argument(
        "--start",
        type=float,
        default=300.0,
        help="Sweep start frequency (Hz), inclusive. Must be >= 100.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=400.0,
        help="Sweep end frequency (Hz), inclusive.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Source FEM case JSON to copy into the lab (default: FEM/configs/guitar_3d.json).",
    )
    parser.add_argument(
        "--mpiexec",
        action="store_true",
        help="Pass --use-mpiexec to fem_master_dynamic (Linux Open MPI workers).",
    )
    parser.add_argument(
        "--cleanup-packaging",
        action="store_true",
        help=(
            "Pass --cleanup to package_rom (removes lab SORTING vectors after ROM write; "
            "also removes VTK/XDMF/H5 under FEM/outputs/modes_3d per package_rom — use with care)."
        ),
    )
    args = parser.parse_args()

    if float(args.start) < 100.0 or float(args.end) < float(args.start):
        print("Error: require --start >= 100 and --end >= --start.", file=sys.stderr)
        return 1

    repo = _repo_root()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labs = repo / "LABS"
    labs.mkdir(parents=True, exist_ok=True)
    lab = labs / f"test_run_{stamp}"
    lab.mkdir(parents=True, exist_ok=False)

    source_config = (
        args.config.resolve()
        if args.config is not None
        else (repo / "FEM" / "configs" / "guitar_3d.json").resolve()
    )
    if not source_config.is_file():
        print(f"Error: config not found: {source_config}", file=sys.stderr)
        return 1

    try:
        lab_config, lab_mesh = _prepare_lab(repo, lab, source_config)
    except (OSError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"Error: lab setup failed: {exc}", file=sys.stderr)
        try:
            shutil.rmtree(lab, ignore_errors=True)
        except Exception:
            pass
        return 1

    py = sys.executable
    master = repo / "FEM" / "scripts" / "fem_master_dynamic.py"
    tuner = repo / "FEM" / "scripts" / "dynamic_filter_tuner.py"
    packer = repo / "FEM" / "scripts" / "package_rom.py"
    for p in (master, tuner, packer):
        if not p.is_file():
            print(f"Error: missing script {p}", file=sys.stderr)
            return 1

    sorting_root = (lab / "SORTING").resolve()
    lab_rom = lab / "rom" / "lab_rom.npz"
    csv_out = sorting_root / "selected_modes.csv"
    plot_out = sorting_root / "selection_plot.png"

    master_cmd = [
        py,
        str(master),
        "--config",
        str(lab_config),
        "--hz-min",
        str(float(args.start)),
        "--hz-max",
        str(float(args.end)),
        "--sorting-root",
        str(sorting_root),
    ]
    if args.mpiexec:
        master_cmd.append("--use-mpiexec")

    if _run_step("Lab Step A — fem_master_dynamic (isolated SORTING)", master_cmd, repo) != 0:
        return 1

    tuner_cmd = [
        py,
        str(tuner),
        "--headless",
        "--candidates",
        str(sorting_root / "candidates_log.json"),
        "--export",
        str(csv_out),
        "--plot-out",
        str(plot_out),
    ]
    if _run_step("Lab Step B — dynamic_filter_tuner (headless)", tuner_cmd, repo) != 0:
        return 1

    pack_cmd = [
        py,
        str(packer),
        "--csv",
        str(csv_out),
        "--out",
        str(lab_rom),
        "--sorting-root",
        str(sorting_root),
    ]
    if args.cleanup_packaging:
        pack_cmd.append("--cleanup")

    if _run_step("Lab Step C — package_rom → lab ROM only", pack_cmd, repo) != 0:
        return 1

    print(
        f"\n{'=' * 72}\n"
        f"  Lab run complete\n"
        f"{'=' * 72}\n"
        f"  Lab directory:     {lab.resolve()}\n"
        f"  Config (copy):     {lab_config}\n"
        f"  Mesh (copy):       {lab_mesh}\n"
        f"  SORTING (private): {sorting_root}\n"
        f"  ROM output:        {lab_rom}\n"
        f"  Plot:              {plot_out}\n"
        f"  Global FEM/SORTING and ROM/classic were not modified by this script.\n"
        f"{'=' * 72}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
