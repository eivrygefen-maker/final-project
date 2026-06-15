#!/usr/bin/env bash
# Lightweight BOX APP/STK path smoke — no STK render, no WAV generation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0
ok() { echo "OK  $1"; PASS=$((PASS + 1)); }
bad() { echo "FAIL $1"; FAIL=$((FAIL + 1)); }

echo "== APP STK BOX smoke =="

for f in \
  tools/generate_box_lhs_pool.py \
  tools/run_app_stk_box_overnight_batch.sh \
  tools/run_app_stk_note_library_box_sample_000.sh
do
  if [[ -f "${REPO_ROOT}/${f}" ]]; then
    ok "present ${f}"
  else
    bad "missing ${f}"
  fi
done

if grep -q 'INSTRUMENT="${INSTRUMENT:-box}"' "${REPO_ROOT}/tools/run_app_stk_box_overnight_batch.sh"; then
  ok "overnight default instrument box"
else
  bad "overnight default instrument box"
fi

if grep -q 'COUNT="${COUNT:-40}"' "${REPO_ROOT}/tools/run_app_stk_box_overnight_batch.sh"; then
  ok "overnight default count 40"
else
  bad "overnight default count 40"
fi

if grep -q 'STK_PARALLEL_WORKERS="${STK_PARALLEL_WORKERS:-3}"' "${REPO_ROOT}/tools/run_app_stk_box_overnight_batch.sh"; then
  ok "overnight default 3 STK workers"
else
  bad "overnight default 3 STK workers"
fi

if grep -q 'parallel_batch' "${REPO_ROOT}/tools/run_app_stk_box_overnight_batch.sh"; then
  ok "overnight parallel_batch render mode"
else
  bad "overnight parallel_batch render mode"
fi

if grep -q 'BOX_NIGHT_ISOLATION_OK' "${REPO_ROOT}/tools/run_app_stk_box_overnight_batch.sh"; then
  ok "overnight isolation marker"
else
  bad "overnight isolation marker"
fi

if grep -q 'app_stk_note_cache/box' "${REPO_ROOT}/tools/run_app_stk_box_overnight_batch.sh"; then
  ok "overnight box cache root"
else
  bad "overnight box cache root"
fi

if grep -q 'debug_reports/box' "${REPO_ROOT}/tools/run_app_stk_box_overnight_batch.sh"; then
  ok "overnight box report root"
else
  bad "overnight box report root"
fi

if grep -q 'build_app_stk_note_library.py' "${REPO_ROOT}/tools/run_app_stk_box_overnight_batch.sh"; then
  ok "overnight calls python build tool directly"
else
  bad "overnight calls python build tool directly"
fi

# classical cache must not be a write target (comparison/assertion substrings allowed)
if grep -E 'app_stk_note_cache/classical|instrument classical' "${REPO_ROOT}/tools/run_app_stk_box_overnight_batch.sh" >/dev/null 2>&1; then
  bad "overnight script references classical cache/instrument as target"
else
  ok "overnight script avoids classical cache target"
fi

python3 - <<'PY'
import sys
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root / "gui"))
sys.path.insert(0, str(root / "tools"))

from app_stk_instrument import (
    default_sample_id,
    instrument_from_shape,
    lhs_pool_path,
    list_lhs_sample_ids,
    rom_shape_namespace,
    shared_shape_name,
)
from stk_app_audio_service import (
    library_report_paths_for_hash,
    list_available_samples,
    preview_cache_dir,
)
from generate_box_lhs_pool import (
    DEFAULT_COUNT,
    box_sample_id,
    build_pool_document,
    is_box_sample_ready,
)

assert rom_shape_namespace("Box") == "box"
assert instrument_from_shape("Box") == "box"
assert default_sample_id("box") == "box_sample_000"
assert shared_shape_name("box") == "box"
assert DEFAULT_COUNT == 40
assert box_sample_id(0) == "box_sample_000"
assert box_sample_id(39) == "box_sample_039"

doc = build_pool_document(count=40, seed=20260616, existing={}, force=False)
assert len(doc["entries"]) == 40
assert doc["entries"][0]["id"] == "box_sample_000"
assert doc["entries"][39]["id"] == "box_sample_039"
assert doc["entries"][0]["parameters"]["geometry.shape_type"] == "box"
assert "lhs_bounds" in doc

lhs = lhs_pool_path(root, "box")
assert lhs.is_file(), lhs
ids = list_lhs_sample_ids(root, "box")
assert "box_sample_000" in ids
box_ids = list_available_samples(root, "box")
assert "box_sample_000" in box_ids
assert "box_sample_000" not in list_available_samples(root, "classical")

preview = preview_cache_dir("deadbeef", "box")
assert "/box/" in preview.as_posix().replace("\\", "/")

report_json, _ = library_report_paths_for_hash("abc123", "box")
assert "/debug_reports/box/" in report_json.as_posix().replace("\\", "/")

status = is_box_sample_ready(root, "box_sample_missing_999")
assert status.get("ready") is False

print("box_path_smoke_ok")
PY
if [[ $? -eq 0 ]]; then ok "python path / LHS dry logic"; else bad "python path / LHS dry logic"; fi

if python3 tools/generate_box_lhs_pool.py --dry-run --count 40 >/dev/null 2>&1; then
  ok "generate_box_lhs_pool dry-run count=40"
else
  bad "generate_box_lhs_pool dry-run count=40"
fi

echo ""
echo "Passed: ${PASS}  Failed: ${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
echo "BOX path smoke complete."
