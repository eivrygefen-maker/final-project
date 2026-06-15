# STK Classical Guitar — APP Integration

## Accepted baseline

The classical guitar STK path is **accepted** (`accepted_for_gui_and_next_shape`).

| Item | Value |
|------|--------|
| Renderer | STK/C++ (`cpp/stk_pgsm_guitar_demo`) |
| Python role | Parameter export only |
| Demo calibration | `v4_10_samples` continuous physical mix |

Python does **not** synthesize guitar body/string audio for this path.

## Final user flow

1. **Save & Sync** — updates GMSH geometry and runs ROM.
2. **After ROM** — STK note cache builds automatically in the background (invisible to the user).
3. **User sees** simple status only:
   - Waiting for guitar simulation
   - Preparing guitar sound…
   - Guitar sound is ready
   - Sound preparation failed — retry Save & Sync
4. **Generate Sound** — saves only when STK cache is **fully ready**; loads the interactive fretboard player.
5. **FIFO comparison stack** — up to **3 fully ready** guitars only (no pending entries).
6. **Load** on a saved guitar switches the fretboard player to that guitar's STK note cache.

Melody support is **future work**.

```
Save & Sync → GMSH + ROM → background STK (quiet)
    → user sees “Preparing guitar sound…”
    → when ready: “Guitar sound is ready”
Generate Sound (ready only) → FIFO stack + activate HTML guitar player
Load saved guitar → switch player cache (no regeneration)
```

## STK background (developer)

- Preview cache: `audio/app_stk_note_cache/classical/current_preview_<hash>/`
- Saved stack cache: `audio/app_stk_note_cache/classical/saved_<guitar_id>/`
- Stack index: `audio/app_stk_guitar_stack/classical/stack_index.json`
- Progress: `audio/debug_reports/app_stk_background_status_<hash>.json`
- Report: `audio/debug_reports/app_stk_note_library_classical_preview_<hash>_report.json`

`refresh_stk_background_job_status()` reconciles subprocess, report, and WAV files. Developer diagnostics are hidden behind **Advanced STK diagnostics** (collapsed by default).

### Internal statuses (not shown to users)

`not_started`, `waiting_for_rom`, `running`, `partial_ready`, `ready`, `failed`, `stale`

`stale` is normal when parameters changed while an old job was running.

## Generate / FIFO rules

- Generate does **not** start STK rendering.
- Generate does **not** save while STK is still running — shows: *Guitar sound is still being prepared. Please wait a little longer.*
- FIFO contains **ready** entries only (`status: ready`).
- Duplicate `parameter_hash` is not added again — existing entry is loaded instead.
- 4th save evicts oldest ready guitar and removes its stack-owned cache directory.

## Guitar player

The HTML fretboard (`components/guitar_player`) plays STK cached WAVs via `activate_stk_guitar_for_player()`:

- Maps fretboard positions to STK note files (E2:E5 chromatic cache)
- Copies WAVs into `components/guitar_player/runtime_cache/<fingerprint>/`
- Player activates after Generate or Load from saved guitars row

Legacy Python synthesis note cache is **not** used for the STK path.

## Priority notes & future speed (background only)

Background jobs render **A2, A4, E5** first, then remaining E2:E5 notes. This is not exposed in the main UX.

Future options (not implemented):

- Worker parallelism for note renders
- Shorter preview duration
- Stronger cache reuse by `parameter_hash`

Full E2:E5 library observed at approximately **270–432 s** on VM hardware.

## Site smoke (VM)

```bash
chmod +x tools/run_app_stk_site_smoke.sh
./tools/run_app_stk_site_smoke.sh
```

Uses isolated `current_preview_smoke_test_<hash>/` fake caches. Verifies ready-only FIFO, duplicate guard, player activation, partial/stale handling. No melody JSON. No 37-note render.

## Preserved validation paths

- `tools/run_app_stk_note_library_classical_sample_000.sh`
- `tools/run_stk_pgsm_demo_v4_10_samples.sh`
- `tools/run_stk_classical_final_acceptance.sh`
