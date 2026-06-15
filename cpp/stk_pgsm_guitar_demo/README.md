# PGSM STK Guitar Demo (C++)

Conference demo renderer: **3 guitars × 3 notes → 9 WAV files**.

Python (`gui/pgsm_stk_parameter_export.py`) exports physical parameters only.
This binary performs STK/C++ synthesis on the VM.

## Physical chain

1. Pluck / contact (`stk::Plucked`)
2. String → bridge force proxy (string derivative × mobility × send)
3. Smoothed bridge drive → body modal bank (bandpass resonators)
4. Top / back / air radiation mix
5. WAV output (`stk::FileWvOut`)

**Important:** the body is driven by bridge force, not a second pluck.

## Build (VM)

```bash
export STK_ROOT=/home/vboxuser/stk   # or your STK install
./tools/build_stk_pgsm_demo.sh
```

CMake searches `STK_ROOT`, then common paths (`~/stk`, `/home/vboxuser/stk`, `/usr/local/stk`).
Fails with a clear message if `Stk.h` or `libstk` is missing.

## Run (VM)

```bash
# 1. Export parameters (Python, no audio)
python gui/pgsm_stk_parameter_export.py

# 2. Render WAVs
./tools/run_stk_pgsm_demo.sh
```

## Outputs

- WAVs: `audio/pgsm_stk_guitar_demo/sample_XXX_{A2|A4|E5}_stk_guitar.wav`
- Reports: `audio/debug_reports/pgsm_stk_guitar_demo_report.json` and `.md`

## Limitations

- Modal bank uses resonator proxies, not live FEM/ROM modes.
- Bridge admittance feedback is simplified.
- Tuned for &lt;1 minute total runtime on the demo set.
