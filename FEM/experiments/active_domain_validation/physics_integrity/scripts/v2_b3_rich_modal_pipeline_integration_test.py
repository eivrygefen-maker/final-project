#!/usr/bin/env python3
"""Lightweight integration checks for rich modal v1 (no PETSc solve)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v2_b3_rich_modal_lib import (  # noqa: E402
    RichModalCollector,
    prolongate_active_to_W,
    frequency_dedupe_report,
    MODES_ACTIVE_NPZ,
    RICH_MODAL_MANIFEST_JSON,
)


def test_prolongate_and_collector_roundtrip() -> None:
    n_active = 5
    n_w = 8
    n_u = 5
    built = {
        "active_local": np.arange(n_active, dtype=np.int32),
        "free_rows": np.arange(n_w, dtype=np.int32),
        "u_idx": np.arange(n_u, dtype=np.int32),
        "p_idx": np.arange(n_u, n_w, dtype=np.int32),
        "n_w": n_w,
    }
    x_active = np.linspace(0.1, 0.5, n_active)
    x_full = prolongate_active_to_W(x_active, built)
    assert x_full.shape == (n_w,)

    coll = RichModalCollector()
    coll.add_mode(
        x_active=x_active,
        target_index=0,
        target_hz=244.0,
        record={
            "frequency_hz": 244.39,
            "lambda_real": 1.0e6,
            "lambda_imag": 0.0,
            "eps_slot_index": 0,
            "eps_compute_error_relative": 1e-9,
            "u_norm_W": 1.0,
            "p_norm_W": 0.5,
            "p_support": 0.33,
            "x_norm_W": 1.5,
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        ckpt = out / "ckpt"
        ckpt.mkdir()
        (ckpt / "synthesis_metadata.json").write_text("{}", encoding="utf-8")
        manifest = coll.write_bundle(
            out / "rich_modal",
            checkpoint_dir=ckpt,
            solve_output_dir=out,
            factor_solver="mkl_pardiso",
            nev=12,
            ncv=24,
            target_set="full9",
            targets_hz=[244.0],
            acceptance_interval_hz=[220.0, 265.0],
            synthesis_metadata_path=ckpt / "synthesis_metadata.json",
        )
        assert manifest["mode_count"] == 1
        assert (out / "rich_modal" / MODES_ACTIVE_NPZ).is_file()
        assert (out / "rich_modal" / RICH_MODAL_MANIFEST_JSON).is_file()

    dedupe = frequency_dedupe_report(
        [{"frequency_hz": 244.0}, {"frequency_hz": 244.05}],
        tol_hz=0.1,
    )
    assert dedupe["duplicate_groups"] >= 1


def main() -> int:
    test_prolongate_and_collector_roundtrip()
    print("[B3_rich_modal_integration_test] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
