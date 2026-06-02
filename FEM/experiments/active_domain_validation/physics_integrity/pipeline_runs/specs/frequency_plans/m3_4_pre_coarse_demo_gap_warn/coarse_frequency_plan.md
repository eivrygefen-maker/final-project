# Coarse frequency plan (M3.4-pre)

- schema: `b3_coarse_frequency_plan_v2`
- mode: `dry-run`
- will_execute: `False`
- calibration_status: `not_calibrated_yet`
- zone_policy_status: `not_calibrated_yet`
- freq_range_hz: `[60.0, 550.0]`
- coarse_step_hz: `15.0`
- target_window_half_width_hz: `1.5` (source: `explicit_override`)
- recommended_target_window_half_width_hz: `7.5`
- coarse_target_count: `34`
- execution_status: `executable_with_discovery_mode_opt_in`

## Regions (placeholder — not calibrated)

- **R_low_60_220** `[60.0, 221.5]` — `not_calibrated_yet`: Low/mid guitar band; requires acceptance-band extension before Stage B discovery
- **R_full9_validated_220_265** `[221.5, 264.0]` — `validated_by_m3_pilot_full9`: M3 m3exec2 timing 9/9 PASS; overlaps solver acceptance band; not the full 60–550 Hz planning range
- **R_mid_high_265_550** `[264.0, 550.0]` — `not_calibrated_yet`: Upper guitar/modal band; requires acceptance-band extension before Stage B discovery

## Spacing alternatives (planning)

| policy | step_hz | targets | note |
|--------|---------|---------|------|

## Cost / parallel guidance

- estimated_wall_minutes: `[68.0, 272.0]`
- parallel (solver): `Run alone on VM: MKL/PETSc ST solves are CPU/RAM heavy on L_prod; do not assume a few minutes for wide-band scans; avoid concurrent solver benchmarks.`

## Recommended next step

Gate A discovery mode is implemented (opt-in). 1) Approve coarse scan plan (15 Hz uniform, exclusive VM). 2) Run density experiment with --B3-discovery-mode on a new output dir. 3) Post-process unique_accepted_frequencies_hz per window; calibrate zones.


## Coverage gap warning

- pair_gap_count: `33`
- max_uncovered_gap_hz: `12.0`

- Planning band 60–550 Hz is the guitar/modal exploration target; 220–265 Hz is validated full9 reference only.
- Zone density thresholds are not_calibrated_yet; regions are placeholders.
- Gate A: opt-in --B3-discovery-mode uses discovery_band + per-target window (Option C).
- Discovery half-width default: effective_coarse_spacing_hz / 2 (touching windows at grid step).
- Default full9 / timing runs without discovery flags keep legacy [220, 265] acceptance.
- Do not overwrite m3exec1/m3exec2 runtime diagnostics.
- WARN: discovery half-width 1.5 Hz < spacing/2 (7.5 Hz); 33 adjacent target pairs leave frequency gaps (max uncovered 12.0 Hz). Use --target-window-half-width-hz >= spacing/2 for coarse discovery.
- WARN: checkpoint_dir not found on this host: C:\projects\final-project\final-project\x