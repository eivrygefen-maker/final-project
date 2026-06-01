#!/usr/bin/env python3
"""Fast smoke test: Stage B rich-modal CLI flag parsing and argv forwarding (no solve)."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_checkpoint_pipeline_lib import (  # noqa: E402
    B3_EXPORT_RICH_MODAL_DATA_ARG,
    build_checkpoint_multi_benchmark_argv,
    parse_rich_modal_export_flag,
)


def _fail(msg: str) -> None:
    print(f"[B3_rich_modal_argv_smoke] FAIL {msg}", flush=True)
    raise SystemExit(1)


def main() -> int:
    off_argv = build_checkpoint_multi_benchmark_argv(
        checkpoint_dir="/tmp/ckpt",
        factor_solver="mkl_pardiso",
        target_set="full9",
        nev=12,
        ncv=24,
        output_dir="/tmp/out",
        export_rich_modal_data=False,
    )
    if parse_rich_modal_export_flag(off_argv):
        _fail("rich modal flag present when export_rich_modal_data=False")
    if B3_EXPORT_RICH_MODAL_DATA_ARG in off_argv:
        _fail(f"{B3_EXPORT_RICH_MODAL_DATA_ARG} in argv when disabled")

    on_argv = build_checkpoint_multi_benchmark_argv(
        checkpoint_dir="/tmp/ckpt",
        factor_solver="mkl_pardiso",
        target_set="full9",
        nev=12,
        ncv=24,
        output_dir="/tmp/out",
        export_rich_modal_data=True,
    )
    if not parse_rich_modal_export_flag(on_argv):
        _fail("rich modal flag missing when export_rich_modal_data=True")
    if on_argv.count(B3_EXPORT_RICH_MODAL_DATA_ARG) != 1:
        _fail(f"expected one {B3_EXPORT_RICH_MODAL_DATA_ARG} in forwarded argv")

    import argparse

    def _rich_flag_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(add_help=False)
        p.add_argument(
            B3_EXPORT_RICH_MODAL_DATA_ARG,
            dest="export_rich_modal_data",
            action="store_true",
            default=False,
        )
        return p

    solve_ns, _ = _rich_flag_parser().parse_known_args(
        [
            "--checkpoint-dir",
            "/tmp/ckpt",
            B3_EXPORT_RICH_MODAL_DATA_ARG,
        ]
    )
    if not solve_ns.export_rich_modal_data:
        _fail("solve-style parser did not set export_rich_modal_data")

    multi_ns, _ = _rich_flag_parser().parse_known_args(
        [
            "--checkpoint-dir",
            "/tmp/ckpt",
            B3_EXPORT_RICH_MODAL_DATA_ARG,
        ]
    )
    if not multi_ns.export_rich_modal_data:
        _fail("multi-style parser did not set export_rich_modal_data")

    forwarded = build_checkpoint_multi_benchmark_argv(
        checkpoint_dir="/tmp/ckpt",
        factor_solver="mkl_pardiso",
        target_set="full9",
        nev=12,
        ncv=24,
        output_dir="/tmp/out",
        export_rich_modal_data=bool(solve_ns.export_rich_modal_data),
    )
    multi_ns2, _ = _rich_flag_parser().parse_known_args(forwarded)
    if not multi_ns2.export_rich_modal_data:
        _fail("forwarded argv did not enable export_rich_modal_data in multi_benchmark")

    print("[B3_rich_modal_argv_smoke] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
