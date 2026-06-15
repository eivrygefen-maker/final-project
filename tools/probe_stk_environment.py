#!/usr/bin/env python3
"""Probe STK install paths for VM setup (read-only, no compile)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _check_root(root: Path) -> dict:
    include = root / "include" / "Stk.h"
    lib_dirs = [root / "build", root / "lib", Path("/usr/local/lib")]
    libs = []
    for d in lib_dirs:
        if not d.is_dir():
            continue
        for name in ("libstk.a", "libstk.so", "stk.lib", "libstk.dylib"):
            p = d / name
            if p.is_file():
                libs.append(str(p))
    rawwaves = root / "rawwaves"
    return {
        "stk_root": str(root),
        "stk_h": include.is_file(),
        "stk_h_path": str(include) if include.is_file() else "",
        "libraries": libs,
        "rawwaves": rawwaves.is_dir(),
        "rawwaves_path": str(rawwaves) if rawwaves.is_dir() else "",
        "ok": include.is_file() and bool(libs),
    }


def main() -> int:
    candidates = []
    env = os.environ.get("STK_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path.home() / "stk",
            Path("/home/vboxuser/stk"),
            Path("/usr/local/stk"),
            Path("/opt/stk"),
        ]
    )
    seen: set[str] = set()
    results = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            results.append(_check_root(c))

    print("PGSM STK environment probe")
    print(f"repo_root: {REPO}")
    if not results:
        print("STATUS: no STK candidate directories found")
        print("Set STK_ROOT to your STK install and re-run.")
        return 1

    ok_any = False
    for r in results:
        status = "OK" if r["ok"] else "INCOMPLETE"
        print(f"\n[{status}] {r['stk_root']}")
        print(f"  Stk.h: {r['stk_h_path'] or 'missing'}")
        print(f"  libs: {r['libraries'] or 'none'}")
        print(f"  rawwaves: {r['rawwaves_path'] or 'missing'}")
        ok_any = ok_any or r["ok"]

    if ok_any:
        print("\nSTATUS: STK ready for build_stk_pgsm_demo.sh")
        return 0
    print("\nSTATUS: STK headers or libstk not found — build will fail until STK is installed.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
