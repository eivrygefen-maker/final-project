#!/usr/bin/env python3
"""One-time: split mixed region_dof_indices.npz into index-only NPZ + region_dof_metadata.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_rich_modal_lib import split_region_dof_mixed_npz_inplace  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split mixed region_dof_indices.npz metadata into region_dof_metadata.json (no A/M rebuild)."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing region_dof_indices.npz (e.g. lprod/checkpoint).",
    )
    args = parser.parse_args(argv)
    result = split_region_dof_mixed_npz_inplace(args.checkpoint_dir.expanduser().resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in ("PASS", "SKIP") else 2


if __name__ == "__main__":
    raise SystemExit(main())
