# CODEX_HANDOFF.md

## Files changed
- `tools/build_classic_stk_contrast_diagnostic.py`
- `CODEX_HANDOFF.md`

## New CLI options
- `--comparison-mode matched_loudness`
- `--comparison-mode bounded_loudness`
- `--comparison-mode raw_level`
- `--comparison-mode all`

## Mode meanings
- `matched_loudness`: full per-clip RMS matching to target, peak ceiling only.
- `bounded_loudness`: current behavior, RMS target with `+/-6 dB` clamp and peak ceiling.
- `raw_level`: no per-clip normalization, one global peak-protection gain on final WAV.

## Outputs
- Each mode writes its own WAV, JSON, and Markdown report.
- Filename includes note and mode, e.g. `classic_contrast_A3_raw_level_<timestamp>.wav`.
- Reports include sample id, cache path, original RMS, gain, final RMS, peak, and decay estimate.
- Reports include RMS/peak/decay spread summaries and a metric-based preliminary conclusion.

## Decision criteria
- A: normalization hides strong differences when `raw_level` is much clearer than `matched_loudness`.
- B: normalization partly hides differences when raw-level spread is reduced but not erased by bounded/matched modes.
- C: normalization is not the main limitation when all modes remain weak or matched/bounded still preserve audible differences.
- If A/B: recommend one bounded contrast adjustment using existing body/send/radiation factors only.
- If C: recommend switching website default from conservative to strong and stop audio tuning.

## Website/audio behavior
- Website default preset remains unchanged.
- No FEM/ROM/STK physics changed.
- No STK rendering or audio generation was run in Codex.
- CLASSIC-only remains unchanged.

## VM commands: A2/A3/E5 all modes
```bash
git pull
python -m py_compile tools/build_classic_stk_contrast_diagnostic.py
python tools/build_classic_stk_contrast_diagnostic.py --note A2 --comparison-mode all --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
python tools/build_classic_stk_contrast_diagnostic.py --note A3 --comparison-mode all --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
python tools/build_classic_stk_contrast_diagnostic.py --note E5 --comparison-mode all --max-samples 20 --duration-s 4.5 --silence-s 0.5 --output-dir audio/diagnostics
```

## Optional experiment cache root
```bash
python tools/build_classic_stk_contrast_diagnostic.py --note A3 --comparison-mode all --cache-root audio/app_stk_note_cache/classical_contrast_aggressive --max-samples 20 --output-dir audio/diagnostics
```

## Lightweight checks run
- `python -m py_compile tools\build_classic_stk_contrast_diagnostic.py`
- `python tools\build_classic_stk_contrast_diagnostic.py --help`
- Empty-cache dry run of `--comparison-mode all`; it wrote three no-audio reports and exited nonzero as expected.
