# STK Classical Guitar — APP Integration

## Accepted baseline

The classical guitar STK path is **accepted** (`accepted_for_gui_and_next_shape`).

| Item | Value |
|------|--------|
| Renderer | STK/C++ (`cpp/stk_pgsm_guitar_demo`) |
| Python role | Parameter export only |
| Demo calibration | `v4_10_samples` continuous physical mix |

Python does **not** synthesize guitar body/string audio for this path.

## Configuration

Central APP STK settings live in `config/app_stk_config.json` (safe defaults if missing):

- Note render durations (default **4.5 s**, longer lows, shorter highs)
- FIFO stack flags (`enable_ready_fifo_stack`, max 3)
- Generate intent auto-load (`enable_generate_intent`)
- Overlapping fretboard playback (`enable_overlapping_playback`)
- Preferred render mode: **batch** (one STK/C++ invocation for all notes)

Melody support is **future work** — no `melodies.json` required.

## Final user flow

1. **Save & Sync** — updates GMSH geometry and runs ROM.
2. **After ROM** — STK note cache builds automatically in the background (invisible to the user).
3. **User sees** simple status only:
   - Waiting for guitar simulation
   - Preparing guitar sound…
   - Guitar sound is ready
   - Sound preparation failed — retry Save & Sync
4. **Generate Sound** — when STK is ready, saves guitar (if FIFO enabled) and loads the interactive fretboard player.
   - If Generate was clicked while STK was still running, the player **auto-loads** when rendering completes (no second click).
5. **FIFO comparison stack** (optional, `enable_ready_fifo_stack`) — up to **3 fully ready** guitars only.
6. **Load** on a saved guitar switches the fretboard player to that guitar's STK note cache.

```
Save & Sync → GMSH + ROM → background STK (quiet)
    → user sees “Preparing guitar sound…”
    → when ready: “Guitar sound is ready”
Generate Sound (or intent while running) → FIFO + activate HTML guitar player when ready
Load saved guitar → switch player cache (no regeneration)
```

## Required note set (fretboard-derived)

**Single source of truth:** `config/classical_guitar_fretboard.json`

Standard classical tuning (low to high string):

| String | Open note |
|--------|-----------|
| S6 (top visual row) | E2 |
| S5 | A2 |
| S4 | D3 |
| S3 | G3 |
| S2 | B3 |
| S1 (bottom visual row) | E4 |

**UI orientation** (HTML player):

- Top string = S6 low E
- Open strings on the **right**
- Higher frets extend **left**
- Each fret adds **one semitone**: `note_at_fret = open_midi + fret`

For **19 frets**, the highest note is **B5** (S1 fret 19 = E4 + 19 semitones).

`gui/classical_guitar_fretboard.py` loads the JSON and builds the map by formula (not manual hardcoding). All APP/HTML/STK paths use this module:

- HTML player payload (`note_name`, `string_visual_order_numbers`, `fretboard` metadata)
- STK required note set and cache spec
- Mapping audits (`audio/debug_reports/app_stk_fretboard_mapping_audit.json`)

The player note library is **not** a fixed E2:E5 chromatic range.

Cache identity (`.cache_spec.json`) includes:

- `parameter_hash`
- required note set
- per-note duration fingerprint
- renderer version (`app_stk_v2_fretboard`)

Old **E2:E5 / 2.5 s** caches are treated as **incompatible** and must rebuild.

Note filenames use **sharp** spelling (`A#4`, not `Bb4`); the player resolves flat aliases.

## STK background (developer)

- Preview cache: `audio/app_stk_note_cache/classical/current_preview_<hash>/`
- Saved stack cache: `audio/app_stk_note_cache/classical/saved_<guitar_id>/`
- Stack index: `audio/app_stk_guitar_stack/classical/stack_index.json`
- Progress: `audio/debug_reports/app_stk_background_status_<hash>.json`
- Report: `audio/debug_reports/app_stk_note_library_classical_preview_<hash>_report.json`
- Mapping audit: `audio/debug_reports/app_stk_note_mapping_audit_<hash>.json`

`refresh_stk_background_job_status()` reconciles subprocess, report, and WAV files. Developer diagnostics are hidden behind **Advanced STK diagnostics** (collapsed by default).

While STK is running or generate intent is pending, the APP uses a lightweight **auto-refresh** (default 12 s) until the player loads.

### Internal statuses (not shown to users)

`not_started`, `waiting_for_rom`, `running`, `partial_ready`, `ready`, `failed`, `stale`

## Generate / FIFO rules

- Generate does **not** start STK rendering.
- Generate stores intent if STK is still running; auto-activates when ready.
- FIFO contains **ready** entries only (`status: ready`) when `enable_ready_fifo_stack` is true.
- Duplicate `parameter_hash` is not added again — existing entry is loaded instead.
- 4th save evicts oldest ready guitar and removes its stack-owned cache directory.
- Player works with FIFO disabled (preview cache activation only).

## Guitar player

The HTML fretboard (`components/guitar_player`) plays STK cached WAVs via `activate_stk_guitar_for_player()`:

- Maps all fretboard positions to STK note files (fretboard-derived set, up to **B5** on 19 frets)
- Copies WAVs into `components/guitar_player/runtime_cache/<fingerprint>/`
- **Overlapping playback** — each click starts a new `AudioBufferSource`; prior notes decay naturally (chords)
- Mapping audit blocks player if cache is incomplete

Legacy Python synthesis note cache is **not** used for the STK path.

## Render performance

Default CLI / background job mode: **`--render-mode batch`** — one parameter JSON with all `renders[]`, one STK subprocess.

Fallback: `--render-mode per_note`

Target runtime: **180 s** (`target_runtime_s` in config). Reports include `achieved_target`.

Priority notes (**A2, A4, E5**) render first in ordering; full fretboard set is always required for player readiness.

## Site smoke (VM)

```bash
chmod +x tools/run_app_stk_site_smoke.sh
./tools/run_app_stk_site_smoke.sh
```

Uses isolated `current_preview_smoke_test_<hash>/` fake caches. Verifies config, fretboard note set, audit, FIFO, duplicate guard, generate intent helpers, overlapping JS flag. No full STK render. No melody JSON.

## Preserved validation paths

- `tools/run_app_stk_note_library_classical_sample_000.sh`
- `tools/run_stk_pgsm_demo_v4_10_samples.sh`
- `tools/run_stk_classical_final_acceptance.sh`
