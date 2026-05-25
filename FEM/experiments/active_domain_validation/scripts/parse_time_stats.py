#!/usr/bin/env python3
"""Parse /usr/bin/time -v output into JSON."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_verbose(text: str) -> dict:
    out: dict = {}
    patterns = {
        "elapsed_seconds": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+(\S+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s+(\d+)",
        "user_seconds": r"User time \(seconds\):\s+([\d.]+)",
        "system_seconds": r"System time \(seconds\):\s+([\d.]+)",
        "percent_cpu": r"Percent of CPU this job got:\s+(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            out[key] = m.group(1)
    if "max_rss_kb" in out:
        try:
            out["max_rss_mb"] = float(out["max_rss_kb"]) / 1024.0
        except (TypeError, ValueError):
            pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = args.verbose.read_text(encoding="utf-8", errors="replace")
    payload = parse_verbose(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
