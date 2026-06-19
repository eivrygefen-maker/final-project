# CODEX_HANDOFF.md

## Files changed
- `tools/build_classic_stk_contrast_diagnostic.py`
- `.gitignore`
- `CODEX_HANDOFF.md`

## Script path and CLI usage
- Script: `tools/build_classic_stk_contrast_diagnostic.py`
- A3 comparison:
```bash
python tools/build_classic_stk_contrast_diagnostic.py --note A3 --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
```
- Optional notes:
```bash
python tools/build_classic_stk_contrast_diagnostic.py --note A2 --max-samples 20 --output-dir audio/diagnostics
python tools/build_classic_stk_contrast_diagnostic.py --note E5 --max-samples 20 --output-dir audio/diagnostics
```

## Cache/render behavior
- Uses existing Classical note-cache WAVs from `audio/app_stk_note_cache/classical`.
- Scans cache directories and includes only caches that already contain the requested note WAV.
- Does not invoke STK, FEM, ROM, Streamlit, or any website generation.
- If notes are missing, the report records them and says VM cache regeneration is required.

## Output paths
- WAV: `audio/diagnostics/classic_contrast_A3_<timestamp>.wav`
- JSON: `audio/diagnostics/classic_contrast_A3_<timestamp>.json`
- Markdown: `audio/diagnostics/classic_contrast_A3_<timestamp>.md`
- `.gitignore` now ignores `audio/diagnostics/`.

## Normalization method
- Each note is trimmed/padded to the requested duration.
- Adds requested silence between clips.
- Applies gentle per-clip RMS normalization to `-20 dBFS`.
- Gain change is clamped to `+/-6 dB`.
- Peak ceiling is `-1 dBFS`.
- No compression, EQ, randomization, synthesis, or artificial timbre changes.

## Missing-note behavior
- Missing note WAVs are not rendered by the diagnostic.
- Missing caches are listed in JSON/Markdown under `missing_notes`.
- If no playable clips exist, no WAV is written and the script exits with status `2`.

## CLASSIC-only confirmation
- This is an offline Classical cache reader only.
- No website behavior changed.
- No BOX/ACOUSTIC UI or behavior was touched.
- No FEM/ROM/STK solver, synthesis, or WAV-generation logic changed.

## Lightweight checks run
- `python -m py_compile tools\build_classic_stk_contrast_diagnostic.py`
- `python tools\build_classic_stk_contrast_diagnostic.py --help`
- `python gui\test_stk_note_library_startup_command.py`

## VM command
```bash
git pull
python tools/build_classic_stk_contrast_diagnostic.py --note A3 --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
```
