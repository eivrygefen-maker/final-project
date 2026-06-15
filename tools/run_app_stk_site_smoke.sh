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

from app_stk_config import load_app_stk_config
from app_stk_fretboard import (
    build_required_note_set_from_fretboard,
    lookup_note,
    normalize_note_name,
    note_to_midi,
    required_notes_cover_high_frets,
    validate_explicit_fretboard_checks,
)
from classical_guitar_fretboard import load_fretboard_config, run_fretboard_mapping_audit
from stk_app_audio_service import (
    ACTIVE_JOB_FILE,
    APP_NOTE_CACHE_ROOT,
    DEBUG_REPORTS,
    GUITAR_STACK_ROOT,
    activate_stk_guitar_for_player,
    build_cache_spec_for_hash,
    cache_is_ready,
    cache_is_ready_for_fretboard,
    cleanup_smoke_test_artifacts,
    compute_parameter_hash,
    find_stack_entry_by_hash,
    library_report_paths_for_hash,
    list_ready_guitar_stack,
    load_guitar_stack,
    refresh_stk_background_job_status,
    run_note_mapping_audit,
    save_guitar_stack,
    save_guitar_to_stack,
    set_active_job,
    smoke_test_cache_dir,
    validate_stk_player_runtime_cache,
    write_cache_spec,
    write_minimal_silent_wav,
)

cfg = load_app_stk_config(root)
assert cfg.get("instrument") == "classical"
assert float(cfg.get("default_duration_s", 0)) >= 4.0

fb_cfg_path = root / "config" / "classical_guitar_fretboard.json"
assert fb_cfg_path.is_file(), "missing classical_guitar_fretboard.json"
fb_cfg = load_fretboard_config(root)
assert fb_cfg.get("tuning", {}).get("S6") == "E2"
explicit = validate_explicit_fretboard_checks(fb_cfg)
assert all(row["passed"] for row in explicit), explicit
assert lookup_note(6, 0, fb_cfg) == "E2"
assert lookup_note(6, 1, fb_cfg) == "F2"
assert lookup_note(6, 2, fb_cfg) == "F#2"
assert lookup_note(6, 3, fb_cfg) == "G2"
assert lookup_note(5, 3, fb_cfg) == "C3"
assert lookup_note(1, 19, fb_cfg) == "B5"

required = build_required_note_set_from_fretboard(int(cfg.get("fret_count") or 19))
assert required, "fretboard required notes empty"
assert note_to_midi(required[-1]) >= note_to_midi("B5"), required
assert normalize_note_name("Bb4") == "A#4"

s1_high = required_notes_cover_high_frets(19, string_number=1, min_fret=13)
s2_high = required_notes_cover_high_frets(19, string_number=2, min_fret=18)
assert s1_high, "S1 frets 13+ missing from mapping"
assert s2_high, "S2 frets 18+ missing from mapping"
for n in s1_high + s2_high:
    assert n in required, f"missing mapped note {n}"

for p in (
    APP_NOTE_CACHE_ROOT / "classical",
    GUITAR_STACK_ROOT / "classical",
    DEBUG_REPORTS,
):
    p.mkdir(parents=True, exist_ok=True)

SMOKE_READY = compute_parameter_hash("smoke_test_ready_rom", {"smoke": "ready"})
SMOKE_PARTIAL = compute_parameter_hash("smoke_test_partial_rom", {"smoke": "partial"})
SMOKE_STALE = compute_parameter_hash("smoke_test_stale_rom", {"smoke": "stale"})
SMOKE_OLD = compute_parameter_hash("smoke_test_old_rom", {"smoke": "old"})

for ph in (SMOKE_READY, SMOKE_PARTIAL, SMOKE_STALE, SMOKE_OLD):
    cleanup_smoke_test_artifacts(ph)

saved_active = None
if ACTIVE_JOB_FILE.is_file():
    saved_active = ACTIVE_JOB_FILE.read_text(encoding="utf-8")

def write_fake_report(ph: str, cache: Path, note_names: list[str], spec_hash: str) -> None:
    report_json, _ = library_report_paths_for_hash(ph)
    report_json.write_text(
        json.dumps(
            {
                "parameter_hash": ph,
                "readiness": "ready_for_app_playback",
                "status": "ready",
                "note_count": len(note_names),
                "fretboard_required_note_count": len(required),
                "cache_spec_hash": spec_hash,
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
spec = build_cache_spec_for_hash(SMOKE_READY, cfg)
write_cache_spec(cache_ready, spec)
for n in required:
    write_minimal_silent_wav(cache_ready / f"{normalize_note_name(n)}.wav", duration_s=0.05)
write_minimal_silent_wav(cache_ready / "all_notes_preview.wav", duration_s=0.05)
write_fake_report(SMOKE_READY, cache_ready, required, spec["cache_spec_hash"])
set_active_job(SMOKE_READY)

state_ready = refresh_stk_background_job_status(
    SMOKE_READY, promote_stack=False, cache_dir=cache_ready
)
assert state_ready["status"] == "ready", state_ready
assert cache_is_ready_for_fretboard(cache_ready, SMOKE_READY)

audit = run_note_mapping_audit(cache_ready, SMOKE_READY)
assert audit["passed"], audit
assert audit.get("all_notes_preview_exists") is True
assert "all_notes_preview.wav" in audit.get("ignored_non_note_wavs", []), audit
assert audit.get("valid_note_wav_count") == len(required), audit
assert lookup_note(6, 1, fb_cfg) == "F2"
assert lookup_note(6, 2, fb_cfg) == "F#2"
assert lookup_note(6, 3, fb_cfg) == "G2"
assert lookup_note(1, 19, fb_cfg) == "B5"

entry = save_guitar_to_stack(
    parameter_hash=SMOKE_READY,
    display_name="Smoke Ready Guitar",
    geometry_summary={"length": 0.5},
    repo_root=root,
)
assert entry["status"] == "ready"
dup = save_guitar_to_stack(parameter_hash=SMOKE_READY, display_name="Dup", repo_root=root)
assert dup.get("_duplicate") is True
assert len(list_ready_guitar_stack()) == 1

activation = activate_stk_guitar_for_player(
    cache_dir=Path(entry["note_cache_path"]),
    parameter_hash=SMOKE_READY,
    saved_guitar_id=entry["saved_guitar_id"],
)
payload = activation["player_payload"]
assert payload.get("enable_overlapping_playback") is True
assert payload.get("string_visual_order_numbers") == [6, 5, 4, 3, 2, 1]
s6f1 = next(p for p in payload["positions"] if p["string"] == 6 and p["fret"] == 1)
assert s6f1.get("note_name") == "F2", s6f1
fb_audit = run_fretboard_mapping_audit(
    cache_dir=Path(entry["note_cache_path"]), player_payload=payload, repo_root=root
)
assert fb_audit.get("readiness") == "ready_fretboard_mapping", fb_audit
validation = validate_stk_player_runtime_cache(
    payload, runtime_dir=Path(activation["runtime_dir"])
)
assert validation["ok"], validation
assert validation.get("preview_wav") == "all_notes_preview.wav"
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
    save_guitar_to_stack(parameter_hash=SMOKE_PARTIAL, display_name="Partial", repo_root=root)
    raise AssertionError("partial save should fail")
except RuntimeError:
    pass

cache_old = smoke_test_cache_dir(SMOKE_OLD)
cache_old.mkdir(parents=True, exist_ok=True)
for n in ("E2", "F2", "G2", "A2", "B2", "C3", "D3", "E3", "F3", "G3", "A3", "B3", "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5"):
    write_minimal_silent_wav(cache_old / f"{n}.wav", duration_s=0.05)
assert not cache_is_ready_for_fretboard(cache_old, SMOKE_OLD), "old E2:E5 cache must be incomplete"
assert cache_is_ready(cache_old, note_range="E2:E5"), "legacy range check still works"

set_active_job(SMOKE_READY)
cache_stale = smoke_test_cache_dir(SMOKE_STALE)
cache_stale.mkdir(parents=True, exist_ok=True)
write_minimal_silent_wav(cache_stale / "A2.wav", duration_s=0.05)
state_stale = refresh_stk_background_job_status(
    SMOKE_STALE, promote_stack=False, cache_dir=cache_stale
)
assert state_stale["status"] == "stale", state_stale
assert find_stack_entry_by_hash(SMOKE_PARTIAL) is None

import stk_app_ui

class _FakeSt:
    session_state = {}

    @staticmethod
    def warning(*_args, **_kwargs):
        pass

stk_app_ui.st = _FakeSt()
stk_app_ui.st.session_state = {
    "stk_generate_intent_hash": SMOKE_READY,
}
from stk_app_ui import fulfill_generate_intent_if_ready

result = fulfill_generate_intent_if_ready(
    repo_root=root,
    rom_fp="smoke_test_ready_rom",
    lhs_params={"smoke": "ready"},
    geom={},
    top_wood="spruce",
    back_wood="mahogany",
)
assert result is not None, "generate intent should auto-load when ready"
assert result.get("action") in ("saved_new", "loaded_existing", "activated_preview")
assert stk_app_ui.st.session_state.get("stk_generate_intent_hash") == ""

for ph in (SMOKE_READY, SMOKE_PARTIAL, SMOKE_STALE, SMOKE_OLD):
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
  ok "B–E. config, fretboard, FIFO, audit, intent, validation"
else
  bad "B–E. refresh/FIFO smoke logic"
fi

if rg -n 'enable_overlapping_playback|overlappingPlayback' gui/components/guitar_player/index.html >/dev/null 2>&1; then
  ok "overlapping playback JS support flag"
else
  bad "overlapping playback JS missing"
fi

if [[ -f "${REPO_ROOT}/config/classical_guitar_fretboard.json" ]]; then
  ok "config/classical_guitar_fretboard.json present"
else
  bad "config/classical_guitar_fretboard.json missing"
fi

if [[ -f "${REPO_ROOT}/config/app_stk_config.json" ]]; then
  ok "config/app_stk_config.json present"
else
  bad "config/app_stk_config.json missing"
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
