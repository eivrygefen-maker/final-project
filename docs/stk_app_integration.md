# STK Classical Guitar — APP Integration

## Accepted baseline

The classical guitar STK path is **accepted** (`accepted_for_gui_and_next_shape`).

| Item | Value |
|------|--------|
| Renderer | STK/C++ (`cpp/stk_pgsm_guitar_demo`) |
| Python role | Parameter export only |
| Demo calibration | `v4_10_samples` continuous physical mix |
| Acceptance report | `audio/debug_reports/pgsm_stk_classical_final_acceptance_report.json` |

Python does **not** synthesize guitar body/string audio for this path.

## User flow (APP)

1. **Save & Sync** — updates GMSH geometry and runs the ROM pipeline (unchanged).
2. **After ROM succeeds** — STK note-library rendering starts automatically in the background (non-blocking).
3. **Status refresh** — on every Streamlit rerun, `refresh_stk_background_job_status()` reconciles subprocess state, progress JSON, library report, and WAV files. When the report says `ready_for_app_playback` with 37 notes, the UI promotes `running` → `ready` even if the job file was stale.
4. **Generate Sound** — save/confirm action (does **not** start STK). If cache is ready, saves immediately with `status: ready`. If STK is still running, saves as `status: pending_audio` and attaches cache automatically when the matching `parameter_hash` completes.
5. **Playback** — uses cached STK note WAVs from preview or saved stack directories. Note list is populated from actual `*.wav` files in the cache directory.

Melody support is **future work** and not part of this step.

## Pipeline

```
User design → Save & Sync → GMSH + ROM
    → parameter_hash = SHA256(rom_fp + lhs_params)[:16]
    → background STK job (subprocess, priority A2/A4/E5 first)
    → gui/pgsm_stk_parameter_export.py (build_render_entry, v4_10_samples)
    → STK parameter JSON (one note per render)
    → stk_pgsm_guitar_demo (C++/STK)
    → audio/app_stk_note_cache/classical/current_preview_<hash>/<Note>.wav
    → Generate Sound → saved_<guitar_id>/ + FIFO stack (ready or pending_audio)
```

## STK job statuses

| Status | Meaning |
|--------|---------|
| `not_started` | No job for current hash |
| `waiting_for_rom` | ROM not ready yet |
| `running` | Background subprocess rendering notes |
| `partial_ready` | Priority notes A2/A4/E5 ready; full E2:E5 still rendering |
| `ready` | Preview cache complete (37 notes E2:E5) |
| `failed` | Renderer or export error |
| `stale` | Design/hash mismatch; result ignored |

FIFO stack entry statuses: `pending_audio`, `ready`, `failed_audio`, `stale`.

Cache hit: if `current_preview_<hash>/` already has all notes, the job is marked ready immediately without rerendering.

## Priority notes

Background jobs render **A2, A4, E5 first**, then the remaining E2:E5 chromatic notes. This improves perceived readiness (~35 s for first useful playback vs ~270–432 s for the full library depending on hardware).

CLI:

```bash
python tools/build_app_stk_note_library.py \
  --priority-notes A2 A4 E5 \
  ...
```

## Runtime

Full E2:E5 chromatic library (37 notes) has been observed at approximately **270–432 s** total render time on VM hardware (~7–12 s per note).

## Generate all notes (VM — reference library)

```bash
./tools/build_stk_pgsm_demo.sh
chmod +x tools/run_app_stk_note_library_classical_sample_000.sh
./tools/run_app_stk_note_library_classical_sample_000.sh
```

Background APP jobs use additional flags:

```bash
python tools/build_app_stk_note_library.py \
  --sample-id sample_000 \
  --cache-dir audio/app_stk_note_cache/classical/current_preview_<hash> \
  --parameter-hash <hash> \
  --job-status-json audio/debug_reports/app_stk_background_job_<hash>.json \
  --priority-notes A2 A4 E5
```

## APP integration modules

| Module | Role |
|--------|------|
| `gui/stk_app_audio_service.py` | `refresh_stk_background_job_status`, priority render order, FIFO pending promotion |
| `gui/stk_app_ui.py` | Streamlit panel (note list, A2/A4/E5 quick play, stack status) |
| `gui/app.py` | ROM → `schedule_stk_after_rom`, refresh on rerun, Generate → stack save |
| `tools/build_app_stk_note_library.py` | CLI / subprocess note library builder |
| `tools/run_app_stk_site_smoke.sh` | Lightweight import/path/refresh smoke |

Streamlit: **Step 3 → expander “STK Classical Guitar (accepted renderer)”**

## What is cached

| Path | Contents |
|------|----------|
| `audio/app_stk_note_cache/classical/current_preview_<hash>/` | Active preview note WAVs |
| `audio/app_stk_note_cache/classical/saved_<guitar_id>/` | Promoted stack-owned caches |
| `audio/app_stk_guitar_stack/classical/stack_index.json` | Last 3 guitar snapshots (FIFO, max 3) |
| `audio/debug_reports/app_stk_background_status_<hash>.json` | Per-note progress during render |
| `audio/debug_reports/app_stk_background_job_<hash>.json` | Job status snapshot |
| `audio/debug_reports/app_stk_note_library_classical_preview_<hash>_report.json` | Final timing/readiness report |

## FIFO comparison stack

- Max **3** saved guitars.
- Generate while STK is running → `pending_audio`; auto-promoted to `ready` when matching hash completes.
- Generate when STK is ready → immediate `ready` with copied cache.
- `parameter_hash` is the authority for matching pending entries to completed caches.

## Site smoke (VM)

```bash
chmod +x tools/run_app_stk_site_smoke.sh
./tools/run_app_stk_site_smoke.sh
```

Uses **isolated** `current_preview_smoke_test_<hash>/` directories and cleans up its own artifacts. Does not depend on real user preview caches, manual site testing leftovers, or melodies. Verifies refresh promotion, partial/stale handling, and FIFO pending→ready promotion without a 37-note render.

### Status notes

- `stale` is normal when parameters changed and an old hash no longer matches the active job.
- `partial_ready` means priority notes (A2/A4/E5) exist but the full E2:E5 library is still rendering.
- Refresh promotes `ready` from a hash's own report + WAV count, independent of which hash is the active session job.

## Preserved validation paths

- `tools/run_app_stk_note_library_classical_sample_000.sh`
- `tools/run_stk_pgsm_demo_v4_10_samples.sh`
- `tools/run_stk_classical_final_acceptance.sh`

## Future work

- Wire user-designed guitar through ROM → physical export (beyond `sample_000` interim)
- Melody library and UI
- Interactive fretboard player using STK note cache
