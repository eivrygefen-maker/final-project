#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _is_number_list(x: Any) -> bool:
    if not isinstance(x, list) or not x:
        return False
    for v in x:
        if not isinstance(v, (int, float)):
            return False
    return True


def find_modes_anywhere(obj: Any) -> list[float] | None:
    """
    Recursively search for a numeric list under common keys:
    - modes_hz
    - modes
    Accepts nested placement (e.g. data/result blocks).
    """
    if isinstance(obj, dict):
        # Try direct keys first
        for key in ("modes_hz", "modes"):
            if key in obj and _is_number_list(obj[key]):
                return [float(v) for v in obj[key]]

        # Otherwise search children
        for v in obj.values():
            out = find_modes_anywhere(v)
            if out is not None:
                return out

    elif isinstance(obj, list):
        for item in obj:
            out = find_modes_anywhere(item)
            if out is not None:
                return out

    return None


def load_modes(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    modes = find_modes_anywhere(data)
    if not modes:
        raise SystemExit(
            f"ERROR: no modes array found anywhere in {path}\n"
            f"Expected a numeric list under 'modes_hz' or 'modes' (possibly nested)."
        )
    return modes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssss", required=True, help="Path to SSSS result JSON")
    ap.add_argument("--cccc", required=True, help="Path to CCCC result JSON")
    ap.add_argument("--top", type=int, default=8, help="How many modes to print")
    args = ap.parse_args()

    ssss = Path(args.ssss)
    cccc = Path(args.cccc)

    ms = load_modes(ssss)
    mc = load_modes(cccc)

    n = min(max(1, args.top), len(ms), len(mc))

    print(f"\nCompare BC (first {n} modes)")
    print(f"SSSS: {ssss}")
    print(f"CCCC: {cccc}\n")
    print(f"{'mode':>4} | {'f_SSSS [Hz]':>12} | {'f_CCCC [Hz]':>12} | {'ratio':>8}")
    print("-" * 46)

    for i in range(n):
        f_s = ms[i]
        f_c = mc[i]
        ratio = (f_c / f_s) if f_s > 1e-12 else float("nan")
        print(f"{i+1:>4} | {f_s:>12.4f} | {f_c:>12.4f} | {ratio:>8.4f}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
