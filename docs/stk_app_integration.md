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

## Pipeline

```
ROM / LHS physical parameters
    → gui/pgsm_stk_parameter_export.py (build_render_entry, v4_10_samples)
    → STK parameter JSON (one note per render)
    → stk_pgsm_guitar_demo (C++/STK)
    → audio/app_stk_note_cache/classical/<sample_id>/<Note>.wav
    → APP playback (Streamlit st.audio / comparison stack)
```

Melodies:

```
audio/app_stk_melody_library/melodies.json  (note names + durations)
    → tools/render_app_stk_melody.py (concatenate cached WAVs + fades)
    → audio/app_stk_melody_cache/classical/<sample_id>/<melody_id>.wav
```

## Generate all notes (VM)

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

- **37 notes** for E2:E5 chromatic
- Cache hits skip STK when WAV exists (unless `--force`)
- Timing report: `audio/debug_reports/app_stk_note_library_classical_sample_000_report.json`

## APP integration modules

| Module | Role |
|--------|------|
| `gui/stk_app_audio_service.py` | Note library build, cache paths, guitar FIFO stack |
| `gui/stk_app_ui.py` | Streamlit panel (generate, play note, melodies, compare) |
| `tools/build_app_stk_note_library.py` | CLI note library builder |
| `tools/render_app_stk_melody.py` | Melody assembly from cache |

Streamlit: **Step 3 → expander “STK Classical Guitar (accepted renderer)”**

## What is cached

| Path | Contents |
|------|----------|
| `audio/app_stk_note_cache/classical/<sample>/` | Per-note STK WAVs (`E2.wav`, …) |
| `audio/app_stk_note_cache/.render_tmp/` | Per-note STK render temp (safe to delete) |
| `audio/app_stk_melody_cache/classical/<sample>/` | Assembled melody WAVs |
| `audio/app_stk_guitar_stack/classical/stack_index.json` | Last 3 guitar snapshots (FIFO) |

## Runtime measurement

The note library report includes:

- `total_render_time_s`
- `average_time_per_note_s`
- `slowest_note` / `fastest_note`
- `cache_hit_count` / `cache_miss_count`

## Preserved validation paths

These remain unchanged:

- `tools/run_stk_pgsm_demo_v4_10_samples.sh`
- `tools/run_stk_classical_final_acceptance.sh`
- v3 demo and reports

## Future work

- Wire user-designed guitar (`website` sample) through ROM → physical export
- Extend note range above E5
- Box-shape STK pipeline (next major step)
- Optional v4.1 mix strengthening (only if audit recommends; not applied automatically)
- Interactive fretboard player using STK note cache instead of Python synthesis cache
