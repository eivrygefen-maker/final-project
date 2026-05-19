#!/usr/bin/env python3
"""
Harmonic resolvent probe for coupled u–p FSI at a single frequency.

Solves (A - omega^2 M) x = F with the same mesh, BCs, and assembled operators as the
main EVP driver, then reports ||u||, ||p||, and ||p||/||u||.

Usage (VM, single MPI rank — same as fem_worker_single):

  mpiexec -n 1 python FEM/scripts/run_resolvent_probe.py \\
    --config FEM/SORTING/pipeline_merged_configs/sample_001.json

  mpiexec -n 1 python FEM/scripts/run_resolvent_probe.py \\
    --config FEM/SORTING/pipeline_merged_configs/sample_001.json \\
    --hz 102 --force-tag 3 --force-scale 1.0

Interpretation:
  - ||p||/||u|| > 1e-6  -> coupling visible; physics OK, focus on eigensolver strategy.
  - ||p||/||u|| ~ 0      -> formulation/interface coupling issue before tuning SLEPc.

Stabilization (probe path only): soft shell grounding, air-volume pressure penalty,
block Frobenius scaling of A/M blocks, symmetric diagonal equilibration, MUMPS shift 1e-2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mpi4py import MPI

SCRIPT_DIR = Path(__file__).resolve().parent
FEM_ROOT = SCRIPT_DIR.parent
REPO_ROOT = FEM_ROOT.parent
DEFAULT_CONFIG = FEM_ROOT / "SORTING" / "pipeline_merged_configs" / "sample_001.json"


def _resolve_mesh_path(cfg: dict, config_path: Path) -> Path:
    raw = Path(cfg["solver"]["mesh_file"])
    if raw.is_absolute():
        return raw
    for base in (config_path.parent, config_path.parents[1], FEM_ROOT, REPO_ROOT):
        cand = (base / raw).resolve()
        if cand.is_file():
            return cand
    return (FEM_ROOT / raw).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Coupled FSI resolvent probe at one frequency (A - w^2 M) x = F."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Merged FEM JSON (same as pipeline / fem_master_dynamic).",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=102.0,
        help="Drive frequency in Hz (default 102).",
    )
    parser.add_argument(
        "--force-tag",
        type=int,
        default=3,
        choices=(1, 3, 4),
        help="Facet tag for harmonic traction: 1=top, 3=back (default), 4=ribs.",
    )
    parser.add_argument(
        "--force-scale",
        type=float,
        default=1.0,
        help="Traction amplitude (N/m^2 scale in weak form units).",
    )
    args = parser.parse_args()

    if MPI.COMM_WORLD.size != 1:
        if MPI.COMM_WORLD.rank == 0:
            print(
                "[resolvent-probe] Requires exactly one MPI rank "
                "(use: mpiexec -n 1 python ...).",
                file=sys.stderr,
            )
        return 2

    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    mesh_path = _resolve_mesh_path(cfg, config_path)
    if not mesh_path.is_file():
        print(
            f"Mesh not found: {mesh_path}\n"
            "Run the pipeline mesh step first (run_pipeline.py / mesh_sync).",
            file=sys.stderr,
        )
        return 1
    cfg.setdefault("solver", {})["mesh_file"] = str(mesh_path)

    if MPI.COMM_WORLD.rank == 0:
        print(f"[resolvent-probe] Config: {config_path}")
        print(f"[resolvent-probe] Mesh:   {mesh_path}")
        print(
            f"[resolvent-probe] f={args.hz} Hz, force_tag={args.force_tag}, "
            f"force_scale={args.force_scale}"
        )
        sys.stdout.flush()

    sys.path.insert(0, str(SCRIPT_DIR))
    from fem_main_3d import run_coupled_resolvent_probe  # noqa: WPS433

    try:
        stats = run_coupled_resolvent_probe(
            cfg,
            frequency_hz=float(args.hz),
            force_facet_tag=int(args.force_tag),
            force_scale=float(args.force_scale),
        )
    except Exception as exc:
        if MPI.COMM_WORLD.rank == 0:
            print(f"[resolvent-probe] FAILED: {exc}", file=sys.stderr)
        return 1

    if MPI.COMM_WORLD.rank == 0:
        out_path = config_path.parent / "resolvent_probe_result.json"
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[resolvent-probe] Wrote {out_path}")
        if not stats.get("solve_ok", False):
            return 2
        return 0 if stats.get("coupling_check_pass") else 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
