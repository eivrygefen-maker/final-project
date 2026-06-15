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
3. **Generate Sound** — does **not** start STK from zero. It saves the current guitar into the FIFO comparison stack when the preview cache is ready.
4. **Playback** — uses cached STK note WAVs from preview or saved stack directories.

Melody support (dropdown, `melodies.json`, melody cache) is **future work** and not part of this step.

## Pipeline

```
User design → Save & Sync → GMSH + ROM
    → parameter_hash = SHA256(rom_fp + lhs_params)[:16]
    → background STK job (subprocess)
    → gui/pgsm_stk_parameter_export.py (build_render_entry, v4_10_samples)
    → STK parameter JSON (one note per render)
    → stk_pgsm_guitar_demo (C++/STK)
    → audio/app_stk_note_cache/classical/current_preview_<hash>/<Note>.wav
    → Generate Sound → promote preview → saved_<guitar_id>/ + FIFO stack
```

## STK job statuses

| Status | Meaning |
|--------|---------|
| `not_started` | No job for current hash |
| `waiting_for_rom` | ROM not ready yet |
| `running` | Background subprocess rendering notes |
| `ready` | Preview cache complete (37 notes E2:E5) |
| `failed` | Renderer or export error |
| `stale` | Design changed while job was running; result ignored |

Cache hit: if `current_preview_<hash>/` already has all notes, the job is marked ready immediately without rerendering.

## Generate all notes (VM — reference library)

```bash
./tools/build_stk_pgsm_demo.sh
chmod +x tools/run_app_stk_note_library_classical_sample_000.sh
./tools/run_app_stk_note_library_classical_sample_000.sh
```

Or directly:

```bash
python tools/build_app_stk_note_library.py \
  --sample-id sample_000 \
  --instrument classical \
  --note-range E2:E5 \
  --output-root audio/app_stk_note_cache
```

Background APP jobs use additional flags:

```bash
python tools/build_app_stk_note_library.py \
  --sample-id sample_000 \
  --cache-dir audio/app_stk_note_cache/classical/current_preview_<hash> \
  --parameter-hash <hash> \
  --job-status-json audio/debug_reports/app_stk_background_job_<hash>.json
```

- **37 notes** for E2:E5 chromatic (~270 s total on reference hardware)
- Cache hits skip STK when WAV exists (unless `--force`)

## APP integration modules

| Module | Role |
|--------|------|
| `gui/stk_app_audio_service.py` | Parameter hash, preview cache, background jobs, FIFO stack |
| `gui/stk_app_ui.py` | Streamlit panel (ROM/STK status, preview playback, comparison stack) |
| `gui/app.py` | Wires ROM completion → `schedule_stk_after_rom`, Generate → stack save |
| `tools/build_app_stk_note_library.py` | CLI / subprocess note library builder |
| `tools/run_app_stk_site_smoke.sh` | Lightweight import/path smoke (no browser, no full render) |

Streamlit: **Step 3 → expander “STK Classical Guitar (accepted renderer)”**

## What is cached

| Path | Contents |
|------|----------|
| `audio/app_stk_note_cache/classical/current_preview_<hash>/` | Active preview note WAVs |
| `audio/app_stk_note_cache/classical/saved_<guitar_id>/` | Promoted stack-owned caches |
| `audio/app_stk_note_cache/classical/sample_000/` | Reference VM library (legacy layout) |
| `audio/app_stk_note_cache/.render_tmp/` | Per-note STK render temp (safe to delete) |
| `audio/app_stk_guitar_stack/classical/stack_index.json` | Last 3 guitar snapshots (FIFO, max 3) |
| `audio/debug_reports/app_stk_background_job_<hash>.json` | Background job progress/status |

## FIFO comparison stack

- Max **3** saved guitars (`stack_index.json`).
- On 4th save: oldest entry removed and its `saved_<id>/` cache deleted.
- Each entry includes: `saved_guitar_id`, `display_name`, `timestamp`, `parameter_hash`, `geometry_summary`, `rom_physical_summary_path`, `note_cache_path`, `timing_report_path`, `source_sample_id`.

## Site smoke (VM)

```bash
chmod +x tools/run_app_stk_site_smoke.sh
./tools/run_app_stk_site_smoke.sh
```

Checks: STK binary/build script, Python imports, optional `sample_000` cache, FIFO stack read/write, creatable cache paths. Does **not** require melodies or run a full 37-note render.

## Preserved validation paths

These remain unchanged:

- `tools/run_app_stk_note_library_classical_sample_000.sh`
- `tools/run_stk_pgsm_demo_v4_10_samples.sh`
- `tools/run_stk_classical_final_acceptance.sh`
- v3 demo and reports

## Future work

- Wire user-designed guitar (`website` sample) through ROM → physical export (beyond `sample_000` interim)
- Melody library and UI
- Extend note range above E5
- Box-shape STK pipeline
- Interactive fretboard player using STK note cache instead of Python synthesis cache
