# M4 ROM output gap analysis

**Date:** 2026-06-05  
**Scope:** FOM vs ROM output parity for STK/audio/website. **No FEM physics changes.**

---

## Executive summary

| Layer | Status |
|-------|--------|
| **M4 FOM** | Rich per-mode scalar catalog (~500–600 deduped modes) |
| **ROM Phase-1** | Frequency-only k-NN surrogate |
| **ROM Phase-2** | Frequencies + lightweight scalars (implemented) |
| **Full mode-shape parity** | Not attempted (Stage C out of scope) |

Validated no-leakage frequency accuracy on `sample_016`: median relative error ≈ **0.85%**. Phase-2 extends ROM toward STK-consumable modal metadata without eigenvectors.

---

## 1. Which FOM fields does ROM currently predict?

### Phase-1 (legacy `m4_modal_surrogate_v1`)

| Field | ROM |
|-------|-----|
| `frequency_hz` | **Yes** (sorted-index k-NN IDW) |
| All other catalog fields | **No** (`null` / `field_availability: false`) |

### Phase-2 (`m4_modal_surrogate_v2`)

| Field | ROM |
|-------|-----|
| `frequency_hz` | **Yes** (sorted-index k-NN IDW) |
| `top_share`, `back_share`, `air_share` | **Yes** (nearest-frequency IDW blend) |
| `coupling_class`, `dominant_region`, `secondary_region` | **Yes** (weighted categorical vote) |
| `bridge_excitation_coupling`, `bridge_excitation_abs` | **Yes** |
| `radiation_proxy`, `mic_output_proxy`, `modal_norm` | **Yes** |
| `top_output_proxy`, `back_output_proxy`, `air_pressure_proxy` | **Yes** (optional) |
| `bridge_excitation_region`, `mic_output_method`, `audio_coupling_status` | **Yes** (optional categorical) |
| Raw participation scores (`top_participation`, …) | **No** (derivable from shares; not primary STK input) |
| `participation_method`, `audio_coupling_detail`, status subfields | **No** (diagnostic) |

---

## 2. Which FOM fields are missing from ROM?

| Category | Missing from ROM |
|----------|------------------|
| **Eigenvectors / mode shapes** | Full displacement vectors, rich modal NPZ |
| **Aggregation provenance** | `chunk_id`, `target_hz`, `zone_id`, `source_worker_result`, … |
| **Raw participation scores** | `top_participation`, `back_participation`, `air_participation` (legacy non-partitioning) |
| **Fine-grained status** | `participation_status`, `bridge_excitation_status`, `mic_output_status`, `modal_norm_method` |
| **Run-level summaries** | `modes_summary.json` aggregates, `runtime_summary.json` wall times |
| **Aggregation QC** | `warnings_and_failures.json`, chunk failure counts |

---

## 3. Required for STK/audio playback

Per [B3_M4_MODE_AUDIO_COUPLING_METADATA.md](B3_M4_MODE_AUDIO_COUPLING_METADATA.md) and [B3_M4_MODE_DOMINANT_REGION_METADATA.md](B3_M4_MODE_DOMINANT_REGION_METADATA.md):

| Priority | Fields | ROM Phase-2 |
|----------|--------|-------------|
| **P0** | `frequency_hz` | Yes |
| **P0** | `top_share`, `back_share`, `air_share` | Yes |
| **P0** | `bridge_excitation_coupling` or `bridge_excitation_abs` | Yes |
| **P0** | `radiation_proxy` or `mic_output_proxy` | Yes |
| **P1** | `modal_norm` | Yes |
| **P1** | `coupling_class` | Yes (classification) |
| **P2** | `dominant_region` | Yes (label; do not use alone for damping) |
| **Not required** | Raw participation scores, chunk provenance | No |

STK can weight damping by shares + route by `coupling_class`; excitation/output by bridge + radiation/mic proxies; amplitude scale by `modal_norm`.

---

## 4. Required for website visualization

| Field | Need | ROM Phase-2 |
|-------|------|-------------|
| `frequency_hz` | Mode spectrum plots | Yes |
| `coupling_class` | Color/marker encoding | Yes |
| `top_share` / `back_share` / `air_share` | Stacked share charts | Yes |
| `radiation_proxy` | Y-axis for radiation plots | Yes |
| `dominant_region` | Legacy marker fallback | Yes |
| `bridge_excitation_coupling` | Bridge coupling scatter | Yes |
| Run-level mode count | Summary cards | From comparison JSON / LHS pool |
| Full mode shapes | 3D deformation view | **No** (future optional; not Phase-2) |

Website can consume `rom_prediction_pre_fom.json` `predicted_modes[]` with the same plot adapters as `modes_catalog.jsonl` (field rename shim only).

---

## 5. Diagnostic-only (not urgent)

| Fields |
|--------|
| `chunk_id`, `target_hz`, `zone_id`, `spacing_hz`, `window_hz` |
| `source`, `source_worker_result`, `source_solver_result` |
| `participation_method`, `participation_detail`, `participation_status` |
| `audio_coupling_method`, `audio_coupling_detail` |
| `bridge_excitation_region`, `mic_output_method` (optional in ROM) |
| `aggregation_result.json` chunk statistics |
| `warnings_and_failures.json` |

---

## 6. Learnable from completed FOM scalar catalogs

**Yes — all Phase-2 fields** are already in `aggregation/modes_catalog.jsonl` per completed sample. No eigenvectors required.

Training source per shape:

```text
ROM/{shape}/lhs_pool.json
pipeline_runs/guitars/<sample>/runs/<run_id>/aggregation/modes_catalog.jsonl
```

---

## 7. Require full mode shapes (do not attempt now)

| Capability | Why shapes needed |
|------------|-------------------|
| True POD/Galerkin ROM | Project operators onto displacement basis |
| Physical microphone pressure | Exterior radiation solve |
| 3D mode deformation GIF/mesh | Eigenvector animation |
| Exact participation recompute | Region DOF projection from vector |
| MMR / vector orthogonality checks | Full `u`/`p` layouts |

---

## 8. Phase-2 fields (implemented)

See `v2_b3_m4_rom_scalar_fields.py` and `m4_modal_surrogate_v2`.

**Required predictions:**

```text
frequency_hz, top_share, back_share, air_share,
coupling_class, dominant_region, secondary_region,
bridge_excitation_coupling, bridge_excitation_abs,
radiation_proxy, mic_output_proxy, modal_norm
```

**Alignment strategy (recommended):**

| Option | Use |
|--------|-----|
| A. Sorted index | Frequencies only (Phase-1) |
| **B. Nearest-frequency per neighbor** | **Phase-2 scalars (implemented)** |
| C. Coupling-class-aware | Future if classification accuracy weak |
| D. Radiation-weighted | Future for audio-weighted training |
| E. Per-band models | Future at 100+ samples |

Frequencies: sorted-index IDW across neighbors (stable for ~600 modes).  
Scalars: for each predicted frequency, find nearest mode in each neighbor catalog, then IDW blend numerics / vote categoricals.

---

## 9. Artifact comparison

### FOM

| File | Content |
|------|---------|
| `modes_catalog.jsonl` | One JSON per deduped mode; full scalar metadata |
| `modes_summary.json` | Counts, share summaries, audio coupling aggregates |
| `runtime_summary.json` | Wall times (~40–60 min/sample) |
| `aggregation_result.json` | Chunk QC, dedupe stats |

### ROM Phase-1

| File | Content |
|------|---------|
| `m4_modal_surrogate.json/npz` | Training manifest + frequency matrices |
| `rom_prediction_pre_fom.json` | `frequencies_hz` only |
| `rom_fom_comparison.json` | Frequency metrics only |

### ROM Phase-2

| File | Content |
|------|---------|
| `m4_modal_surrogate.json` | Schema v2, field lists, alignment method |
| `m4_modal_surrogate.npz` | Frequencies + scalar numeric/cat arrays |
| `rom_prediction_pre_fom.json` | Schema `m4_rom_prediction_v2`, `predicted_modes[]` |
| `rom_fom_comparison.json` | Schema v3: frequency + scalar + classification + runtime metrics |

---

## 10. Shape isolation

| Shape | Paths |
|-------|-------|
| `classic` | `ROM/classic/lhs_pool.json`, `m4_modal_surrogate.*`, `comparisons/` |
| Future `{shape}` | `ROM/{shape}/...` |

Training collects only samples from the pool’s `shape_name`. No cross-shape training unless explicitly requested.

Shared export:

```text
/media/sf_gmar/{shape_name}/plots/
/media/sf_gmar/{shape_name}/summaries/
```

---

## 11. Runtime expectations

| Stage | Typical |
|-------|---------|
| M4 FOM full sample | 40–60 minutes |
| ROM prediction | **< 1–5 seconds** |
| ROM comparison | **< 1–2 seconds** |

Recorded in:

```text
rom_prediction_runtime_s
rom_comparison_runtime_s
total_rom_runtime_s
```

---

## 12. Accuracy targets (tracking only)

| Category | Metric | Target |
|----------|--------|--------|
| Frequency | `median_relative_error` | ≤ 0.05 |
| Shares | `top_share_mae` | TBD (soft 0.10) |
| Radiation | `radiation_proxy_relative_error_median` | TBD (soft 0.25) |
| Mic | `mic_output_proxy_relative_error_median` | TBD (soft 0.25) |
| Classification | `coupling_class_accuracy` | TBD (soft 0.70) |

Non-frequency metrics are **tracked but do not fail** ROM in production.

---

## 13. Workflow after Phase-2

```bash
# Rebuild v2 surrogate (includes scalar fields from catalogs)
python FEM/.../build_m4_rom_from_completed_fom.py \
  --lhs-json ROM/classic/lhs_pool.json --completed-only --max-samples 16

# No-leakage validation
python FEM/.../run_m4_rom_compare.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --force-sample sample_016 \
  --exclude-target-from-training
```

Review:

```text
ROM/classic/comparisons/rom_accuracy_summary.json
  → frequency_accuracy
  → stk_audio_scalar_accuracy
  → classification_accuracy
  → runtime
```

---

## 14. What remains unchanged

- M4 FOM solver, aggregation, participation/audio coupling compute
- No Stage C / full mode-shape storage
- ROM non-blocking in production pipeline
- Legacy `ROMManager.compare()` / POD path separate
