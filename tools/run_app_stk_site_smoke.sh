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

if rg -n 'st\.audio\([^)]*key=' gui >/dev/null 2>&1; then
  bad "st.audio key= still present in gui/"
else
  ok "no st.audio key= in gui/"
fi

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
from stk_app_ui import request_generate_guitar  # noqa: F401
print("imports_ok")
PY
if [[ $? -eq 0 ]]; then ok "A. import checks"; else bad "A. Python import failed"; fi

python3 - <<'PY'
import json
import sys
from pathlib import Path

root = Path(".").resolve()
sys.path.insert(0, str(root / "gui"))

from stk_app_audio_service import (
    ACTIVE_JOB_FILE,
    APP_NOTE_CACHE_ROOT,
    DEBUG_REPORTS,
    GUITAR_STACK_ROOT,
    activate_stk_guitar_for_player,
    cleanup_smoke_test_artifacts,
    compute_parameter_hash,
    find_stack_entry_by_hash,
    library_report_paths_for_hash,
    list_ready_guitar_stack,
    load_guitar_stack,
    refresh_stk_background_job_status,
    save_guitar_stack,
    save_guitar_to_stack,
    set_active_job,
    smoke_test_cache_dir,
    validate_stk_player_runtime_cache,
    write_minimal_silent_wav,
)

for p in (
    APP_NOTE_CACHE_ROOT / "classical",
    GUITAR_STACK_ROOT / "classical",
    DEBUG_REPORTS,
):
    p.mkdir(parents=True, exist_ok=True)

SMOKE_READY = compute_parameter_hash("smoke_test_ready_rom", {"smoke": "ready"})
SMOKE_PARTIAL = compute_parameter_hash("smoke_test_partial_rom", {"smoke": "partial"})
SMOKE_STALE = compute_parameter_hash("smoke_test_stale_rom", {"smoke": "stale"})

for ph in (SMOKE_READY, SMOKE_PARTIAL, SMOKE_STALE):
    cleanup_smoke_test_artifacts(ph)

saved_active = None
if ACTIVE_JOB_FILE.is_file():
    saved_active = ACTIVE_JOB_FILE.read_text(encoding="utf-8")

def write_fake_report(ph: str, cache: Path, note_names: list[str]) -> None:
    report_json, _ = library_report_paths_for_hash(ph)
    report_json.write_text(
        json.dumps(
            {
                "parameter_hash": ph,
                "readiness": "ready_for_app_playback",
                "status": "ready",
                "note_count": len(note_names),
                "output_dir": str(cache).replace("\\", "/"),
                "report_json": str(report_json).replace("\\", "/"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

doc = load_guitar_stack("classical")
doc["snapshots"] = []
save_guitar_stack(doc, "classical")

cache_ready = smoke_test_cache_dir(SMOKE_READY)
cache_ready.mkdir(parents=True, exist_ok=True)
ready_notes = ["E2", "E3", "A2", "A4", "E5", "G3", "B3"]
for n in ready_notes:
    write_minimal_silent_wav(cache_ready / f"{n}.wav", duration_s=0.05)
write_fake_report(SMOKE_READY, cache_ready, ready_notes)
set_active_job(SMOKE_READY)

state_ready = refresh_stk_background_job_status(
    SMOKE_READY, promote_stack=False, cache_dir=cache_ready
)
assert state_ready["status"] == "ready", state_ready

entry = save_guitar_to_stack(
    parameter_hash=SMOKE_READY,
    display_name="Smoke Ready Guitar",
    geometry_summary={"length": 0.5},
)
assert entry["status"] == "ready"
dup = save_guitar_to_stack(parameter_hash=SMOKE_READY, display_name="Dup")
assert dup.get("_duplicate") is True
assert len(list_ready_guitar_stack()) == 1

activation = activate_stk_guitar_for_player(
    cache_dir=Path(entry["note_cache_path"]),
    parameter_hash=SMOKE_READY,
    saved_guitar_id=entry["saved_guitar_id"],
)
payload = activation["player_payload"]
validation = validate_stk_player_runtime_cache(
    payload, runtime_dir=Path(activation["runtime_dir"])
)
assert validation["ok"], validation
runtime = Path(activation["runtime_dir"])
assert (runtime / "all_notes_preview.wav").is_file()
for pos in payload.get("positions") or []:
    assert (runtime / pos["wav"]).is_file(), pos["wav"]
assert validation["position_count"] > 0

cache_partial = smoke_test_cache_dir(SMOKE_PARTIAL)
cache_partial.mkdir(parents=True, exist_ok=True)
for n in ("A2", "A4", "E5"):
    write_minimal_silent_wav(cache_partial / f"{n}.wav", duration_s=0.05)
set_active_job(SMOKE_PARTIAL)
state_partial = refresh_stk_background_job_status(
    SMOKE_PARTIAL, promote_stack=False, cache_dir=cache_partial
)
assert state_partial["status"] in ("partial_ready", "running"), state_partial
try:
    save_guitar_to_stack(parameter_hash=SMOKE_PARTIAL, display_name="Partial")
    raise AssertionError("partial save should fail")
except RuntimeError:
    pass

set_active_job(SMOKE_READY)
cache_stale = smoke_test_cache_dir(SMOKE_STALE)
cache_stale.mkdir(parents=True, exist_ok=True)
write_minimal_silent_wav(cache_stale / "A2.wav", duration_s=0.05)
state_stale = refresh_stk_background_job_status(
    SMOKE_STALE, promote_stack=False, cache_dir=cache_stale
)
assert state_stale["status"] == "stale", state_stale
assert find_stack_entry_by_hash(SMOKE_PARTIAL) is None

for ph in (SMOKE_READY, SMOKE_PARTIAL, SMOKE_STALE):
    cleanup_smoke_test_artifacts(ph)
doc = load_guitar_stack("classical")
doc["snapshots"] = []
save_guitar_stack(doc, "classical")
if saved_active is not None:
    ACTIVE_JOB_FILE.write_text(saved_active, encoding="utf-8")
elif ACTIVE_JOB_FILE.is_file():
    ACTIVE_JOB_FILE.unlink()

print("smoke_logic_ok")
PY
if [[ $? -eq 0 ]]; then
  ok "B–E. FIFO, runtime cache, validation, duplicate guard"
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
