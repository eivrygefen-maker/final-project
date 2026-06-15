#!/usr/bin/env bash
# Lightweight APP STK site smoke — imports and paths only (no browser, no full render).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0

ok() { echo "OK  $1"; PASS=$((PASS + 1)); }
bad() { echo "FAIL $1"; FAIL=$((FAIL + 1)); }

echo "== APP STK site smoke =="

BINARY="${REPO_ROOT}/cpp/stk_pgsm_guitar_demo/build/stk_pgsm_guitar_demo"
BUILD_SCRIPT="${REPO_ROOT}/tools/build_stk_pgsm_demo.sh"
if [[ -x "${BINARY}" ]]; then
  ok "STK renderer binary exists: ${BINARY}"
elif [[ -f "${BUILD_SCRIPT}" ]]; then
  ok "STK build script present (binary not built yet): ${BUILD_SCRIPT}"
else
  bad "STK renderer binary and build script missing"
fi

python3 - <<'PY'
import sys
from pathlib import Path
root = Path(".").resolve()
sys.path.insert(0, str(root / "gui"))
import stk_app_audio_service as svc  # noqa: F401
import stk_app_ui  # noqa: F401
print("imports_ok")
PY
if [[ $? -eq 0 ]]; then ok "note cache service + APP STK UI import"; else bad "Python import failed"; fi

SAMPLE_CACHE="${REPO_ROOT}/audio/app_stk_note_cache/classical/sample_000"
if [[ -d "${SAMPLE_CACHE}" ]]; then
  NOTE_COUNT="$(find "${SAMPLE_CACHE}" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${NOTE_COUNT}" -ge 1 ]]; then
    ok "sample_000 note cache detected (${NOTE_COUNT} WAVs)"
  else
    bad "sample_000 cache dir exists but has no WAVs"
  fi
else
  ok "sample_000 cache not present (optional on dev machine)"
fi

python3 - <<'PY'
import sys
from pathlib import Path
root = Path(".").resolve()
sys.path.insert(0, str(root / "gui"))
from stk_app_audio_service import (
    APP_NOTE_CACHE_ROOT,
    GUITAR_STACK_ROOT,
    load_guitar_stack,
    save_guitar_stack,
    preview_cache_dir,
    compute_parameter_hash,
)

root = Path(".").resolve()
for p in (
    APP_NOTE_CACHE_ROOT / "classical",
    GUITAR_STACK_ROOT / "classical",
    root / "audio" / "debug_reports",
):
    p.mkdir(parents=True, exist_ok=True)

ph = compute_parameter_hash("smoke_rom_fp", {"L": 0.5})
preview = preview_cache_dir(ph)
preview.mkdir(parents=True, exist_ok=True)

doc = load_guitar_stack("classical")
doc.setdefault("max_snapshots", 3)
doc.setdefault("snapshots", [])
save_guitar_stack(doc, "classical")
loaded = load_guitar_stack("classical")
assert "snapshots" in loaded
stack_path = GUITAR_STACK_ROOT / "classical" / "stack_index.json"
assert stack_path.is_file()
print("stack_ok")
PY
if [[ $? -eq 0 ]]; then ok "FIFO stack metadata create/read"; else bad "FIFO stack metadata"; fi

MELODY_JSON="${REPO_ROOT}/audio/app_stk_melody_library/melodies.json"
if [[ -f "${MELODY_JSON}" ]]; then
  ok "melody JSON present (not required for this smoke)"
else
  ok "melody JSON not required"
fi

echo ""
echo "Passed: ${PASS}  Failed: ${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
echo "Smoke complete."
