#!/usr/bin/env bash
# Lightweight BOX APP/STK path smoke — no STK render, no WAV generation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

python3 - <<'PY'
import sys
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root / "gui"))

from app_stk_instrument import (
    default_sample_id,
    instrument_from_shape,
    lhs_pool_path,
    list_lhs_sample_ids,
    rom_shape_namespace,
    shared_shape_name,
)
from stk_app_audio_service import (
    instrument_debug_reports_root,
    library_report_paths_for_hash,
    list_available_samples,
    preview_cache_dir,
)

assert rom_shape_namespace("Box") == "box"
assert rom_shape_namespace("Classical") == "classic"
assert instrument_from_shape("Box") == "box"
assert instrument_from_shape("Classical") == "classical"
assert default_sample_id("box") == "box_sample_000"
assert default_sample_id("classical") == "sample_000"
assert shared_shape_name("box") == "box"
assert shared_shape_name("classical") == "classic"

lhs = lhs_pool_path(root, "box")
assert lhs.is_file(), lhs
ids = list_lhs_sample_ids(root, "box")
assert "box_sample_000" in ids, ids
classic_ids = list_available_samples(root, "classical")
box_ids = list_available_samples(root, "box")
assert "sample_000" in classic_ids
assert "box_sample_000" in box_ids
assert "box_sample_000" not in classic_ids

preview = preview_cache_dir("deadbeef", "box")
assert "/box/" in preview.as_posix().replace("\\", "/")
assert preview.as_posix().replace("\\", "/").endswith("current_preview_deadbeef")

report_json, report_md = library_report_paths_for_hash("abc123", "box")
assert "/debug_reports/box/" in report_json.as_posix().replace("\\", "/")
assert "box" in report_json.name

classic_report_json, _ = library_report_paths_for_hash("abc123", "classical")
assert "/debug_reports/box/" not in classic_report_json.as_posix().replace("\\", "/")

print("box_path_smoke_ok")
PY

echo "BOX path smoke complete."
