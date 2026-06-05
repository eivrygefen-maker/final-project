# M4 mode audio/ROM/STK coupling metadata (lightweight)

Per accepted ST mode, the worker solve path records **scalar** ROM/STK/audio coupling metrics while the accepted mode vector is already in memory. This is production metadata for modal reduction, damping/excitation weighting, and audible-output estimation — **not** full mode-shape storage.

## What this is / is not

| Yes | No |
|-----|-----|
| Small scalars per mode | Full eigenvector arrays |
| Bridge excitation proxy | True bridge patch geometry (unless `u_idx_bridge` added later) |
| Top/back/air output proxies | External microphone radiation solve |
| `mic_output_proxy` with explicit method | Physical microphone pressure |
| `modal_norm` for cross-mode scaling | Mass-normalized mode shapes |

`mic_output_proxy` is a **proxy** unless a true exterior mic sample point exists in the mesh. Methods are recorded in `mic_output_method`.

## Per-mode fields

### Bridge / string excitation

| Field | Meaning |
|-------|---------|
| `bridge_excitation_coupling` | Mean signed top/bridge displacement ÷ `modal_norm` |
| `bridge_excitation_abs` | RMS top/bridge displacement ÷ `modal_norm` |
| `bridge_excitation_region` | `bridge_mask` or `top_plate_proxy` (no dedicated bridge mask yet) |
| `bridge_excitation_status` | `computed`, `proxy`, `not_available` |

### Output / radiation proxies

Normalized by `modal_norm` (‖x‖₂ on full W layout):

| Field | Meaning |
|-------|---------|
| `top_output_proxy` | RMS top-plate displacement |
| `back_output_proxy` | RMS back (+ ribs when `back_includes_ribs`) |
| `air_pressure_proxy` | Max \|p\| in cavity (fallback: RMS) |
| `radiation_proxy` | Weighted blend: 45% top + 15% back + 40% air (renormalized over available) |

### Microphone output proxy

| Field | Meaning |
|-------|---------|
| `mic_output_proxy` | Best available audible-output scalar |
| `mic_output_method` | e.g. `soundhole_displacement_rms_proxy_v1`, `cavity_pressure_max_proxy_v1`, `radiation_proxy_blend_v1` |
| `mic_output_status` | `computed`, `proxy`, `partial`, `not_available` |

Priority: soundhole displacement → cavity pressure → radiation blend.

### Modal normalization

| Field | Meaning |
|-------|---------|
| `modal_norm` | ‖x‖₂ on full W layout |
| `modal_norm_method` | `full_W_l2_norm_v1` |

### Status envelope

| Field | Meaning |
|-------|---------|
| `audio_coupling_status` | `computed`, `partial`, `not_available` |
| `audio_coupling_method` | `lightweight_modal_coupling_v1` |
| `audio_coupling_detail` | e.g. `operator_build_context` or missing-data reason |

## Inputs used

From checkpoint / operator build (no extra FEM session):

- `region_dof_indices.npz`: `u_idx_top`, `u_idx_back`, `u_idx_ribs`, `u_idx_soundhole`, `p_idx_air`
- Future: `u_idx_bridge` when a bridge mask is exported
- `built_metadata.json`: `u_idx`, `p_idx`, `free_rows`, …

## Where it is written

| Artifact | Path |
|----------|------|
| Solver | `worker_results/<chunk>/solver_result.json` → `targets[].accepted_modes[]` |
| Worker | `worker_results/<chunk>/worker_result.json` → `accepted_mode_records[]` |
| Aggregation | `aggregation/modes_catalog.jsonl`, `modes_summary.json`, `runtime_summary.json` |

## Aggregation summary fields

`modes_summary.json` includes:

- `audio_coupling_computed_count`
- `bridge_coupling_available_count`
- `mic_proxy_available_count`
- `radiation_proxy_summary`
- `modal_norm_summary`
- `audio_coupling_summary` (full breakdown)
- `stk_rom_guidance`

## STK / ROM usage (with participation shares)

Future STK/audio synthesis should combine:

1. `frequency_hz`
2. Damping weights: `top_share`, `back_share`, `air_share` (not hard `dominant_region` alone)
3. Excitation: `bridge_excitation_coupling` (or `bridge_excitation_abs`)
4. Output: `radiation_proxy` or `mic_output_proxy`
5. Amplitude scale: `modal_norm`

## Implementation

- Module: `scripts/v2_b3_mode_audio_coupling.py`
- Solve hook: `collect_accepted_st_modes()` in `v2_b3_st_sinvert_solver_lib.py`
- Aggregation: `merge_audio_coupling_into_catalog_record()` in `v2_b3_m4_aggregate_worker_results.py`

Re-running workers on existing checkpoints refreshes coupling fields (no checkpoint re-export required). Re-aggregation alone preserves fields already in `solver_result.json`.

No Stage C, rich modal export, or waveform generation.
