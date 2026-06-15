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
# v1 (baseline)
python gui/pgsm_stk_parameter_export.py
./tools/run_stk_pgsm_demo.sh

# v2 (physical difference audit + factor activation)
python gui/pgsm_stk_parameter_export.py --demo-version v2
./tools/run_stk_pgsm_demo_v2.sh

# v3 (stronger perceptual differentiation)
python gui/pgsm_stk_parameter_export.py --demo-version v3
./tools/run_stk_pgsm_demo_v3.sh
```

## Outputs

- v1 WAVs: `audio/pgsm_stk_guitar_demo/`
- v2 WAVs: `audio/pgsm_stk_guitar_demo_v2/`
- v2 reports: `audio/debug_reports/pgsm_stk_guitar_demo_v2_report.json` and `.md`
- v3 WAVs: `audio/pgsm_stk_guitar_demo_v3/`
- v3 reports: `audio/debug_reports/pgsm_stk_guitar_demo_v3_report.json` and `.md`

## Limitations

- Modal bank uses resonator proxies, not live FEM/ROM modes.
- Bridge admittance feedback is simplified.
- Tuned for &lt;1 minute total runtime on the demo set.
