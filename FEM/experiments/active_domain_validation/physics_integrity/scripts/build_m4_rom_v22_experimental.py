#!/usr/bin/env python3
"""Build experimental v2.2 ROM artifact (does not overwrite production v2.1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_rom_intensity_v21_v22 import experimental_v22_dir  # noqa: E402
from v2_b3_m4_lhs_pool_bridge import load_lhs_pool  # noqa: E402
from v2_b3_m4_modal_surrogate_lib import (  # noqa: E402
    build_surrogate_from_training_rows,
    collect_completed_fom_training_rows,
)
from v2_b3_m4_rom_intensity_v22 import (  # noqa: E402
    MODEL_VERSION_V2_2,
    PREDICTION_METHOD_V2_2,
    SURROGATE_SCHEMA_V2_2,
)
from v2_b3_m4_worker_run_lib import detect_repo_root, rel, utc_now  # noqa: E402
from v2_b3_petsc_util import write_json_atomic  # noqa: E402

SURROGATE_JSON_NAME = "m4_modal_surrogate_v22_experimental.json"
SURROGATE_NPZ_NAME = "m4_modal_surrogate_v22_experimental.npz"


def save_experimental_surrogate(repo_root: Path, model: dict, shape_name: str) -> dict:
    out_dir = experimental_v22_dir(repo_root, shape_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = dict(model["arrays"])
    json_body = {k: v for k, v in model.items() if k != "arrays"}
    json_body["schema"] = SURROGATE_SCHEMA_V2_2
    json_body["model_version"] = MODEL_VERSION_V2_2
    json_body["method"] = PREDICTION_METHOD_V2_2
    json_body["experimental"] = True
    json_body["production_model_preserved"] = True

    json_path = out_dir / SURROGATE_JSON_NAME
    npz_path = out_dir / SURROGATE_NPZ_NAME
    write_json_atomic(json_path, json_body)

    npz_payload = {
        "feature_matrix_norm": np.asarray(arrays["feature_matrix_norm"], dtype=np.float64),
        "frequencies": np.asarray(arrays["frequencies"], dtype=np.float64),
        "mode_counts": np.asarray(arrays["mode_counts"], dtype=np.int32),
        "feature_mean": np.asarray(arrays["feature_mean"], dtype=np.float64),
        "feature_std": np.asarray(arrays["feature_std"], dtype=np.float64),
        "k_neighbors": np.array([int(model.get("k_neighbors") or 5)], dtype=np.int32),
    }
    for key, val in arrays.items():
        if key.startswith(("scalar__", "cat__")):
            npz_payload[key] = np.asarray(val)
    np.savez(npz_path, **npz_payload)

    manifest = {
        "schema": "m4_rom_experimental_v22_manifest_v1",
        "generated_utc": utc_now(),
        "shape_name": shape_name,
        "model_version": MODEL_VERSION_V2_2,
        "prediction_method": PREDICTION_METHOD_V2_2,
        "active_backend": "m4_surrogate_v22_experimental",
        "surrogate_json": SURROGATE_JSON_NAME,
        "surrogate_npz": SURROGATE_NPZ_NAME,
        "production_files_untouched": [
            "ROM/classic/m4_modal_surrogate.json",
            "ROM/classic/m4_modal_surrogate.npz",
            "ROM/classic/rom_model_manifest.json",
        ],
        "training_sample_count": model.get("training_sample_count"),
    }
    write_json_atomic(out_dir / "model_manifest.json", manifest)
    report = {
        "schema": "m4_rom_build_report_v2_2_experimental",
        "generated_utc": utc_now(),
        "shape_name": shape_name,
        "model_version": MODEL_VERSION_V2_2,
        "training_sample_count": model.get("training_sample_count"),
        "outputs": {
            "json": rel(json_path, repo_root=repo_root),
            "npz": rel(npz_path, repo_root=repo_root),
            "manifest": rel(out_dir / "model_manifest.json", repo_root=repo_root),
        },
    }
    write_json_atomic(out_dir / "build_report.json", report)
    return {"json": json_path, "npz": npz_path, "manifest": out_dir / "model_manifest.json", "report": out_dir / "build_report.json"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhs-json", type=Path, required=True)
    parser.add_argument("--completed-only", action="store_true", default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k-neighbors", type=int, default=5)
    args = parser.parse_args()

    repo_root = detect_repo_root(SCRIPT_DIR)
    lhs_path = args.lhs_json if args.lhs_json.is_absolute() else repo_root / args.lhs_json
    pool = load_lhs_pool(lhs_path)
    shape = str(pool.get("shape_name") or "classic")

    training, skipped = collect_completed_fom_training_rows(
        repo_root=repo_root,
        pool=pool,
        completed_only=bool(args.completed_only),
        max_samples=args.max_samples,
    )
    if not training:
        print("error: no training rows", file=sys.stderr)
        return 2

    model = build_surrogate_from_training_rows(
        shape_name=shape,
        training_rows=training,
        k_neighbors=int(args.k_neighbors),
    )
    paths = save_experimental_surrogate(repo_root, model, shape)
    print(f"experimental_json={rel(paths['json'], repo_root=repo_root)}")
    print(f"experimental_npz={rel(paths['npz'], repo_root=repo_root)}")
    print("production v2.1 files NOT modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
