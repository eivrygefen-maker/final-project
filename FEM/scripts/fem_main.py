# FEM/scripts/fem_main.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from fem_api import run_fem_case, FemResult
from fem_config import load_case_json, save_result_json


def fem_result_to_payload(res: FemResult, meta: Dict[str, Any]) -> Dict[str, Any]:
    # stable output schema (easy for STK/python to read later)
    result_block: Dict[str, Any] = {
        "f0_hz": float(res.f0),
        "modes_hz": [float(x) for x in res.modes],
        "n_modes": int(len(res.modes)),
    }

    # NEW: optional mode weights + pickup point (only if provided by solver)
    if getattr(res, "mode_weights", None) is not None:
        result_block["mode_weights"] = [float(x) for x in (res.mode_weights or [])]

    if getattr(res, "pickup_xy", None) is not None:
        # store as [x0, y0] for JSON friendliness
        x0, y0 = res.pickup_xy  # type: ignore[misc]
        result_block["pickup_xy"] = [float(x0), float(y0)]

    return {
        "meta": meta,
        "result": result_block,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="FEM main runner (numeric pipeline).")
    parser.add_argument(
        "--solution-type",
        type=str,
        default="",
        help="Optional solution preset name (from FEM/solutions/solution_types.json).",
    )
    parser.add_argument(
        "--solution-types-file",
        type=str,
        default="FEM/solutions/solution_types.json",
        help="Path to solution types library JSON",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to FEM case JSON (e.g. FEM/configs/rect_plate_rosewood.json).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path. Default goes to FEM/outputs/<config-name>_result.json",
    )
    args = parser.parse_args()

    if not args.config:
        raise SystemExit(
            "Missing --config. Example:\n"
            "  python3 FEM/scripts/fem_main.py --config FEM/configs/rect_plate_rosewood.json"
        )

    config_path = Path(args.config).resolve()
    case = load_case_json(
        config_path,
        solution_type = args.solution_type,
        solution_types_file = args.solution_types_file,
    )

    # Run (dispatcher will call the numeric solver if implemented)
    res = run_fem_case(case)

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        stem = config_path.stem
        out_path = Path("FEM/outputs") / f"{stem}_result.json"

    payload = fem_result_to_payload(
        res,
        meta={
            "config_path": str(config_path),
            "solver": {
                "dimension": int(case.solver.dimension),   # type: ignore[union-attr]
                "model": str(case.solver.model),           # type: ignore[union-attr]
                "n_modes_req": int(case.solver.n_modes),   # type: ignore[union-attr]
                "mesh_size": float(case.solver.mesh_size)  # type: ignore[union-attr]
            },
        },
    )
    save_result_json(out_path, payload)

    print("=== FEM run finished ===")
    print(f"Config: {config_path}")
    print(f"Saved result JSON: {out_path}")
    print(f"f0 = {res.f0:.2f} Hz, n_modes = {len(res.modes)}")
    print("First 4 modes (Hz):", ", ".join(f"{x:.2f}" for x in res.modes[:4]))

    # NEW: print pickup + first weights (for quick sanity check)
    if getattr(res, "mode_weights", None) is not None:
        w = res.mode_weights or []
        print(f"Pickup (x,y) [m]: {getattr(res, 'pickup_xy', None)}")
        print("First 6 mode weights:", ", ".join(f"{x:.4f}" for x in w[:6]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
