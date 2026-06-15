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
from stk_app_ui import try_save_current_guitar_to_stack  # noqa: F401
print("imports_ok")
PY
if [[ $? -eq 0 ]]; then ok "note cache service + APP STK UI + save imports"; else bad "Python import failed"; fi

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
import json
import sys
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root / "gui"))

from stk_app_audio_service import (
    APP_NOTE_CACHE_ROOT,
    DEBUG_REPORTS,
    GUITAR_STACK_ROOT,
    background_status_path,
    compute_parameter_hash,
    count_wavs_in_cache,
    library_report_paths_for_hash,
    list_available_notes,
    load_guitar_stack,
    preview_cache_dir,
    promote_pending_stack_entries,
    refresh_stk_background_job_status,
    save_guitar_stack,
    save_guitar_to_stack,
)

for p in (
    APP_NOTE_CACHE_ROOT / "classical",
    GUITAR_STACK_ROOT / "classical",
    DEBUG_REPORTS,
):
    p.mkdir(parents=True, exist_ok=True)

ph_ready = compute_parameter_hash("smoke_ready_rom", {"L": 0.42})
preview_ready = preview_cache_dir(ph_ready)
preview_ready.mkdir(parents=True, exist_ok=True)
notes = ["E2", "E3", "A2", "A4", "E5"]
for n in notes:
    (preview_ready / f"{n}.wav").write_bytes(b"RIFF")

report_json, _ = library_report_paths_for_hash(ph_ready)
report_doc = {
    "readiness": "ready_for_app_playback",
    "status": "ready",
    "note_count": len(notes),
    "output_dir": str(preview_ready).replace("\\", "/"),
    "report_json": str(report_json).replace("\\", "/"),
}
report_json.write_text(json.dumps(report_doc, indent=2) + "\n", encoding="utf-8")
write_bg = background_status_path(ph_ready)
write_bg.write_text(
    json.dumps(
        {
            "parameter_hash": ph_ready,
            "status": "running",
            "rendered_notes": 2,
            "total_notes": 37,
            "output_dir": str(preview_ready),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

state = refresh_stk_background_job_status(ph_ready, promote_stack=False)
assert state["status"] == "ready", state
assert state["preview_cache_ready"] is True
assert state["latest_report_path"]

listed = list_available_notes("sample_000", cache_dir=preview_ready)
assert "A2" in listed and "E5" in listed

entry_ready = save_guitar_to_stack(
    parameter_hash=ph_ready,
    display_name="Smoke Ready Guitar",
    geometry_summary={"length": 0.5},
)
assert entry_ready["status"] == "ready"
assert entry_ready.get("note_cache_path")

ph_pending = compute_parameter_hash("smoke_pending_rom", {"L": 0.43})
state_pending = refresh_stk_background_job_status(ph_pending, promote_stack=False)
entry_pending = save_guitar_to_stack(
    parameter_hash=ph_pending,
    display_name="Smoke Pending Guitar",
    geometry_summary={"length": 0.51},
)
assert entry_pending["status"] == "pending_audio"

# Promote pending when cache completes
preview_pending = preview_cache_dir(ph_pending)
preview_pending.mkdir(parents=True, exist_ok=True)
for n in notes:
    (preview_pending / f"{n}.wav").write_bytes(b"RIFF")
rep_pending_json, _ = library_report_paths_for_hash(ph_pending)
rep_pending_json.write_text(
    json.dumps(
        {
            "readiness": "ready_for_app_playback",
            "status": "ready",
            "note_count": len(notes),
            "output_dir": str(preview_pending).replace("\\", "/"),
            "report_json": str(rep_pending_json).replace("\\", "/"),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

promoted = promote_pending_stack_entries(ph_pending)
assert promoted, "expected pending promotion"
assert promoted[0]["status"] == "ready"
assert promoted[0].get("note_cache_path")

doc = load_guitar_stack("classical")
doc["snapshots"] = []
save_guitar_stack(doc, "classical")
assert (GUITAR_STACK_ROOT / "classical" / "stack_index.json").is_file()
print("refresh_and_fifo_ok")
PY
if [[ $? -eq 0 ]]; then ok "refresh promotes report to ready; FIFO ready/pending/promote"; else bad "refresh/FIFO smoke logic"; fi

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
