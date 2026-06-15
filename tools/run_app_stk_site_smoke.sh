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
if [[ $? -eq 0 ]]; then ok "A. import checks"; else bad "A. Python import failed"; fi

SAMPLE_CACHE="${REPO_ROOT}/audio/app_stk_note_cache/classical/sample_000"
if [[ -d "${SAMPLE_CACHE}" ]]; then
  NOTE_COUNT="$(find "${SAMPLE_CACHE}" -maxdepth 1 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${NOTE_COUNT}" -ge 1 ]]; then
    ok "sample_000 note cache detected (${NOTE_COUNT} WAVs, optional)"
  else
    ok "sample_000 cache dir exists (no WAVs, optional)"
  fi
else
  ok "sample_000 cache not present (optional)"
fi

python3 - <<'PY'
import json
import shutil
import sys
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root / "gui"))

from stk_app_audio_service import (
    ACTIVE_JOB_FILE,
    APP_NOTE_CACHE_ROOT,
    DEBUG_REPORTS,
    GUITAR_STACK_ROOT,
    background_status_path,
    cleanup_smoke_test_artifacts,
    compute_parameter_hash,
    job_status_path,
    library_report_paths_for_hash,
    list_available_notes,
    load_guitar_stack,
    preview_cache_dir,
    promote_pending_stack_entries,
    refresh_stk_background_job_status,
    save_guitar_stack,
    save_guitar_to_stack,
    set_active_job,
    smoke_test_cache_dir,
)

for p in (
    APP_NOTE_CACHE_ROOT / "classical",
    GUITAR_STACK_ROOT / "classical",
    DEBUG_REPORTS,
):
    p.mkdir(parents=True, exist_ok=True)

# B. FIFO metadata create/read
doc = load_guitar_stack("classical")
doc.setdefault("max_snapshots", 3)
doc.setdefault("snapshots", [])
save_guitar_stack(doc, "classical")
assert (GUITAR_STACK_ROOT / "classical" / "stack_index.json").is_file()

SMOKE_READY = compute_parameter_hash("smoke_test_ready_rom", {"smoke": "ready"})
SMOKE_PARTIAL = compute_parameter_hash("smoke_test_partial_rom", {"smoke": "partial"})
SMOKE_STALE = compute_parameter_hash("smoke_test_stale_rom", {"smoke": "stale"})
SMOKE_PENDING = compute_parameter_hash("smoke_test_pending_rom", {"smoke": "pending"})

for ph in (SMOKE_READY, SMOKE_PARTIAL, SMOKE_STALE, SMOKE_PENDING):
    cleanup_smoke_test_artifacts(ph)

saved_active = None
if ACTIVE_JOB_FILE.is_file():
    saved_active = ACTIVE_JOB_FILE.read_text(encoding="utf-8")

def write_fake_report(ph: str, cache: Path, note_names: list[str]) -> Path:
    report_json, _ = library_report_paths_for_hash(ph)
    report_doc = {
        "parameter_hash": ph,
        "readiness": "ready_for_app_playback",
        "status": "ready",
        "note_count": len(note_names),
        "output_dir": str(cache).replace("\\", "/"),
        "report_json": str(report_json).replace("\\", "/"),
    }
    report_json.write_text(json.dumps(report_doc, indent=2) + "\n", encoding="utf-8")
    return report_json

# C. Ready promotion (isolated smoke cache + matching report)
cache_ready = smoke_test_cache_dir(SMOKE_READY)
cache_ready.mkdir(parents=True, exist_ok=True)
ready_notes = ["E2", "E3", "A2", "A4", "E5"]
for n in ready_notes:
    (cache_ready / f"{n}.wav").write_bytes(b"RIFF")
write_fake_report(SMOKE_READY, cache_ready, ready_notes)
background_status_path(SMOKE_READY).write_text(
    json.dumps(
        {
            "parameter_hash": SMOKE_READY,
            "status": "running",
            "rendered_notes": 2,
            "total_notes": 37,
            "output_dir": str(cache_ready),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
set_active_job("unrelated_active_hash_for_smoke")

state_ready = refresh_stk_background_job_status(
    SMOKE_READY, promote_stack=False, cache_dir=cache_ready
)
assert state_ready["status"] == "ready", state_ready
assert state_ready["preview_cache_ready"] is True
assert state_ready["actual_wav_count"] == len(ready_notes)

listed = list_available_notes("sample_000", cache_dir=cache_ready)
assert "A2" in listed and "E5" in listed

entry_ready = save_guitar_to_stack(
    parameter_hash=SMOKE_READY,
    display_name="Smoke Ready Guitar",
    geometry_summary={"length": 0.5},
)
assert entry_ready["status"] == "ready"
assert entry_ready.get("note_cache_path")

# D. Partial + stale handling
cache_partial = smoke_test_cache_dir(SMOKE_PARTIAL)
cache_partial.mkdir(parents=True, exist_ok=True)
for n in ("A2", "A4", "E5"):
    (cache_partial / f"{n}.wav").write_bytes(b"RIFF")
set_active_job(SMOKE_PARTIAL)
background_status_path(SMOKE_PARTIAL).write_text(
    json.dumps(
        {
            "parameter_hash": SMOKE_PARTIAL,
            "status": "running",
            "rendered_notes": 1,
            "total_notes": 37,
            "output_dir": str(cache_partial),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
state_partial = refresh_stk_background_job_status(
    SMOKE_PARTIAL, promote_stack=False, cache_dir=cache_partial
)
assert state_partial["status"] in ("partial_ready", "running"), state_partial
assert state_partial["status"] != "stale", state_partial
assert state_partial.get("priority_notes_ready") or state_partial["actual_wav_count"] >= 3

set_active_job(SMOKE_READY)
cache_stale = smoke_test_cache_dir(SMOKE_STALE)
cache_stale.mkdir(parents=True, exist_ok=True)
(cache_stale / "A2.wav").write_bytes(b"RIFF")
state_stale = refresh_stk_background_job_status(
    SMOKE_STALE, promote_stack=False, cache_dir=cache_stale
)
assert state_stale["status"] == "stale", state_stale
assert state_stale.get("stale_reason")

# E. FIFO pending → ready promotion
set_active_job(SMOKE_PENDING)
entry_pending = save_guitar_to_stack(
    parameter_hash=SMOKE_PENDING,
    display_name="Smoke Pending Guitar",
    geometry_summary={"length": 0.51},
)
assert entry_pending["status"] == "pending_audio"

cache_pending = smoke_test_cache_dir(SMOKE_PENDING)
cache_pending.mkdir(parents=True, exist_ok=True)
for n in ready_notes:
    (cache_pending / f"{n}.wav").write_bytes(b"RIFF")
write_fake_report(SMOKE_PENDING, cache_pending, ready_notes)

promoted = promote_pending_stack_entries(SMOKE_PENDING)
assert promoted, "expected pending promotion"
assert promoted[0]["status"] == "ready"
assert promoted[0].get("note_cache_path")

# Cleanup smoke-only artifacts; restore active job file
for ph in (SMOKE_READY, SMOKE_PARTIAL, SMOKE_STALE, SMOKE_PENDING):
    cleanup_smoke_test_artifacts(ph)
doc = load_guitar_stack("classical")
doc["snapshots"] = [s for s in doc.get("snapshots", []) if not str(s.get("display_name", "")).startswith("Smoke")]
save_guitar_stack(doc, "classical")
if saved_active is not None:
    ACTIVE_JOB_FILE.write_text(saved_active, encoding="utf-8")
elif ACTIVE_JOB_FILE.is_file():
    ACTIVE_JOB_FILE.unlink()

print("smoke_logic_ok")
PY
if [[ $? -eq 0 ]]; then
  ok "B–E. FIFO, ready, partial, stale, pending promotion"
else
  bad "B–E. refresh/FIFO smoke logic"
fi

MELODY_JSON="${REPO_ROOT}/audio/app_stk_melody_library/melodies.json"
if [[ -f "${MELODY_JSON}" ]]; then
  ok "melody JSON present (not required)"
else
  ok "melody JSON not required"
fi

echo ""
echo "Passed: ${PASS}  Failed: ${FAIL}"
if [[ "${FAIL}" -gt 0 ]]; then
  exit 1
fi
echo "Smoke complete."
