#!/usr/bin/env python3
"""
Minimal structural-only shell diagnostic (no FSI, no pressure gauge, no ribs clamp).

Uses the same orthotropic facet-shell forms as the coupled model on facet tags 1+3 only.

Usage (VM, mpiexec -n 1 recommended):
  python FEM/scripts/run_structural_diagnostic.py
  python FEM/scripts/run_structural_diagnostic.py --config FEM/configs/guitar_3d.json
  python FEM/scripts/run_structural_diagnostic.py --clamp-ribs   # A/B: enable ribs u=0 on coupled only
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BASE = REPO_ROOT / "FEM" / "configs" / "guitar_3d.json"
DEFAULT_OUT = REPO_ROOT / "FEM" / "configs" / "guitar_3d_structural_diag.json"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Structural-only shell diagnostic runner.")
    parser.add_argument("--config", type=Path, default=DEFAULT_BASE, help="Base FEM JSON")
    parser.add_argument("--write-config", type=Path, default=DEFAULT_OUT, help="Merged diagnostic JSON path")
    parser.add_argument(
        "--clamp-ribs",
        action="store_true",
        help="If set, only affects coupled runs; structural branch never clamps ribs.",
    )
    args = parser.parse_args()

    base_path = args.config.resolve()
    if not base_path.is_file():
        print(f"Config not found: {base_path}", file=sys.stderr)
        return 1

    with open(base_path, encoding="utf-8") as f:
        cfg = json.load(f)

    override = {
        "solver": {
            "couple_fluid": False,
            "structural_only_diagnosis": True,
            "clamp_ribs": bool(args.clamp_ribs),
            "structural_shell_facet_tags": [1, 3],
            "structural_only_num_modes": 10,
            "structural_shift_target_hz": 120.0,
            "structural_min_mode_hz": 20.0,
            "structural_expected_hz_min": 80.0,
            "structural_expected_hz_max": 200.0,
            "structural_vacuum_air_dummy": False,
            "adaptive_mode_sifter": False,
            "num_modes": 10,
        }
    }
    merged = _deep_merge(cfg, override)
    out_path = args.write_config.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4)

    print(f"[structural-diag] Wrote {out_path}")
    print(
        "[structural-diag] couple_fluid=False, facet tags [1,3], "
        "free–free (ribs clamp OFF in structural branch)."
    )
    sys.stdout.flush()

    sys.path.insert(0, str(SCRIPT_DIR))
    from fem_main_3d import run_fem_3d_simulation  # noqa: WPS433

    try:
        result = run_fem_3d_simulation(str(out_path))
        print(f"[structural-diag] Done -> {result}")
        return 0
    except Exception as exc:
        print(f"[structural-diag] FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
