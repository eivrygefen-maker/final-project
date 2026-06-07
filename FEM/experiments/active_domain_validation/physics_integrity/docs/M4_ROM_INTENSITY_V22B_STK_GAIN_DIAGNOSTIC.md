# M4 ROM Intensity v2.2b — STK Combined-Gain Diagnostic

**Status:** Experimental / read-only — run on VM to populate numeric results.  
**Scope:** Five-sample holdout LOO (`sample_000`, `sample_005`, `sample_013`, `sample_024`, `sample_027`).  
**Does not modify:** production ROM v2.1, FOM physics, FOM reruns, or production pipeline.

---

## Executive summary

| Item | Conclusion |
|------|------------|
| **Combined-gain formula** | `bridge_excitation_abs × mic_output_proxy` is **mathematically valid** as a dimensionless bilinear coupling proxy; eigenvector-scale invariant; does **not** double-apply `modal_norm` to the same quantity |
| **Signed variant** | `bridge_excitation_coupling × mic_output_proxy` is valid but **signed**; less suitable as a single STK loudness target without phase/sign handling |
| **Mass-normalized alternative** | `bridge × output / modal_mass` is **not recomputable** from current artifacts (`modal_mass` absent; `modal_norm` is L2 not mass) |
| **v2.1 baseline (30 LOO)** | Mic-only remains the strongest continuous target (p95 MAE ≈ 0.233, rank ≈ 0.782, top-20% ≈ 0.471) |
| **v2.2 lesson** | Class/band-aware matching did not improve rank metrics; combined-gain hypothesis is untested until VM run |
| **Preliminary STK recommendation** | **Keep mic-only** until v2.2b numbers beat thresholds; combined gain is a valid *derived diagnostic* but not yet promoted |

**Improvement gate (meaningful):** p95 norm MAE ≤ 0.15–0.18, or rank ≥ 0.85, or top-20% overlap ≥ 0.60.

---

## Task 1 — Amplitude semantics audit

Source: `v2_b3_mode_audio_coupling.py` (`compute_audio_coupling_from_active`).

### Per-field semantics

| Field | Definition | Signed? | Normalization | Eigenvector scale |
|-------|------------|---------|---------------|-------------------|
| `modal_norm` | L2 norm of full W-layout eigenvector `x` | No | Raw norm (not divided) | Linear: doubles when `x → 2x` |
| `bridge_excitation_coupling` | `mean_signed(bridge_or_top_dof) / modal_norm` | **Yes** | Divided once by `modal_norm` | **Invariant** to `x → αx` |
| `bridge_excitation_abs` | `RMS(bridge_or_top_dof) / modal_norm` | No | Divided once by `modal_norm` | **Invariant** |
| `mic_output_proxy` | `soundhole_RMS/modal_norm`, else `cavity_pressure/modal_norm`, else `radiation_proxy` | No | Divided once by `modal_norm` | **Invariant** |
| `radiation_proxy` | Weighted blend of `top/back/air` proxies; each already `/ modal_norm` | No | Divided once per component | **Invariant** |

### Combined quantities

| Quantity | Formula | Units / semantics | Valid? |
|----------|---------|-------------------|--------|
| `bridge_to_mic_gain_raw` | `bridge_excitation_abs × mic_output_proxy` | Dimensionless bilinear coupling: `(RMS_bridge/‖x‖) × (RMS_out/‖x‖)` | **Yes** — combines two distinct spatial projections |
| `bridge_to_radiation_gain_raw` | `bridge_excitation_abs × radiation_proxy` | Same structure for radiation path | **Yes** |
| `signed_bridge_to_mic_gain_raw` | `bridge_excitation_coupling × mic_output_proxy` | Signed excitation × unsigned output | **Valid proxy**, but signed; STK loudness usually wants unsigned |

### Why multiplication is not “double normalization”

Each factor divides a **different** spatial projection by the **same** `modal_norm`:

```text
gain ∝ RMS(bridge DOFs) × RMS(output DOFs) / modal_norm²
```

Under `x → αx`: both RMS terms scale as `α`, `modal_norm` scales as `α`, so `gain` is unchanged. This is **not** re-dividing the same raw DOF amplitude twice.

### Signed vs unsigned for STK

- **Unsigned (`bridge_excitation_abs`)** matches STK-relevant “how loud” when excitation magnitude matters regardless of phase sign.
- **Signed (`bridge_excitation_coupling`)** preserves excitation polarity; useful for directional coupling, not for a single monotonic loudness ranking without `abs()`.

**Diagnostic uses unsigned products** for STK gain targets; signed variant is computed but not primary.

---

## Task 6 — Proxy stability / eigenvector normalization

| Question | Answer |
|----------|--------|
| Do `bridge_excitation_abs`, `mic_output_proxy`, `radiation_proxy` include `/ modal_norm`? | **Yes** — each is a norm-ratio at FOM solve time |
| Does multiplying them double-apply normalization? | **No** — two distinct projections share one norm divisor |
| Is `bridge × mic` eigenvector-scale invariant? | **Yes** |
| Better invariant `bridge × output / modal_mass`? | Theoretically cleaner for mass-orthonormal modes, but **`modal_mass` is not in catalog** |
| Recomputable without FOM rerun? | **Combined unsigned product: yes** from existing scalars. **Mass-normalized: no** |
| Required components in artifacts? | `bridge_excitation_abs`, `mic_output_proxy`, `radiation_proxy`, `modal_norm` — all present in `modes_catalog.jsonl` |

**Promotion gate:** Combined gain is numerically consistent as a derived proxy. Promote to STK target only if LOO metrics beat mic-only on the gates below.

---

## Task 2 — Derived STK targets

Implemented in `v2_b3_m4_rom_stk_gain_targets.py` → `enrich_catalog_stk_gains()`.

| Target | Computation |
|--------|-------------|
| `bridge_to_mic_gain_raw` | `bridge_excitation_abs × mic_output_proxy` |
| `bridge_to_radiation_gain_raw` | `bridge_excitation_abs × radiation_proxy` |
| `*_log10` | `log10(value + ε)`, `ε = 1e-12` |
| `*_p95_norm` | `value / p95_band`, p95 = 95th percentile within guitar, **60–550 Hz** band |
| `*_strength_class` | Quintile labels within guitar: `very_weak`, `weak`, `medium`, `strong`, `very_strong` |

Same band, epsilon, and p95 conventions as ROM Intensity v2.1.

---

## Task 3 — Prediction targets compared

Per holdout, five target families:

### Separate fields (v2.1 baseline path)

- `mic_output_proxy_p95_norm` — **mic only**
- `radiation_proxy_p95_norm` — **radiation only**
- `bridge_excitation_abs_p95_norm` — **bridge only**

### Combined fields

- `bridge_to_mic_gain_p95_norm` — **bridge × mic**
- `bridge_to_radiation_gain_p95_norm` — **bridge × radiation**

### Ranking-only (method D)

- Percentile rank within guitar (60–550 Hz) for mic, radiation, bridge×mic, bridge×radiation

### Top-k (derived from continuous ranks)

- Top 10%, 20%, 30% overlap / precision / recall / NDCG

### Five-level strength (method E)

- `*_strength_class` accuracy vs FOM quintile labels

---

## Task 4 — Prediction methods

| ID | Method | Description |
|----|--------|-------------|
| **A** | `A_v21_separate_idw` | Current v2.1: nearest-frequency IDW on separate fields; combined gain = product of blended factors |
| **B** | `B_combined_nearest_freq_idw` | Direct IDW on `bridge_to_*_gain_raw` with nearest-frequency matching |
| **C** | `C_combined_physics_aware_idw` | Direct IDW on combined raw with class/region/band-aware mode selection (v2.2 matcher) |
| **D** | `D_rank_percentile_idw` | IDW on within-guitar percentile rank |
| **E** | `E_strength_class_vote` | Categorical vote on five-level strength class |

Frequency prediction unchanged (v2.1 sorted-index IDW). No full 30-sample comparison in this diagnostic.

---

## Task 5 — Metrics

Per target × method × holdout:

| Metric | Notes |
|--------|-------|
| `p95_norm_mae` | Primary continuous error |
| `log_mae` | Log-space error |
| `spearman` | Rank correlation on raw values (or rank-percentile for D) |
| `kendall_tau` | If SciPy available |
| `top_10/20/30pct_overlap` | Set overlap on strongest modes |
| `ndcg_at_*` | Ranking quality |
| `strength_class_accuracy` | Method E only |
| `runtime_s` | Per-method wall time |

### STK relevance comparison axes

| Role | Field |
|------|-------|
| Mic only | `mic_output_proxy_p95_norm` |
| Bridge × mic | `bridge_to_mic_gain_p95_norm` |
| Radiation only | `radiation_proxy_p95_norm` |
| Bridge × radiation | `bridge_to_radiation_gain_p95_norm` |

---

## Reference baseline — v2.1 (30-sample LOO)

From validated production-intensity LOO (not v2.2b):

| Metric | Mic only | Radiation only | Bridge only |
|--------|----------|----------------|-------------|
| p95 norm MAE | **0.233** | 0.253 | 0.278 |
| Spearman rank | **0.782** | 0.164 | — |
| Top-20% overlap | **0.471** | 0.235 | — |
| Frequency median error | 1.42% (all methods) | | |

v2.2 class/band matching: mic MAE slightly better; radiation and bridge worse; rank unchanged.

---

## Five-sample holdout results

**Workspace note:** Local checkout has `ROM/classic/lhs_pool.json` but **no** `pipeline_runs/guitars/*/aggregation/modes_catalog.jsonl`. Numeric results require VM execution.

After VM run, read:

```text
ROM/classic/experimental_v22b/diagnostic_summary.json
ROM/classic/experimental_v22b/per_sample/sample_{000,005,013,024,027}.json
```

### Aggregate table (populate from `aggregate_by_method`)

| Method | Target (STK role) | p95 MAE median | Spearman median | Top-20% median |
|--------|-------------------|----------------|-----------------|----------------|
| A | mic only | *VM* | *VM* | *VM* |
| A | bridge × mic | *VM* | *VM* | *VM* |
| A | radiation only | *VM* | *VM* | *VM* |
| A | bridge × radiation | *VM* | *VM* | *VM* |
| B | bridge × mic | *VM* | *VM* | *VM* |
| B | bridge × radiation | *VM* | *VM* | *VM* |
| C | bridge × mic | *VM* | *VM* | *VM* |
| C | bridge × radiation | *VM* | *VM* | *VM* |
| D | mic / combined ranks | *VM* | *VM* | *VM* |
| E | strength classes | *VM* | accuracy *VM* | — |

### Leakage assertions (built into script)

- Target excluded from `build_holdout_surrogate_model(exclude_sample_ids=[target])`
- `target ∉ neighbor_sample_ids` per prediction call
- `training_includes_target: false` in output JSON

---

## Task 8 — Decision logic

Evaluate after VM run against gates and v2.1 baseline:

| Code | Condition | Action if true |
|------|-----------|----------------|
| **A** | Combined p95 MAE ≤ 0.15–0.18 **and** rank/top-k beat mic-only | Adopt **bridge × mic** as STK target |
| **B** | Mic-only remains best on rank and top-k | **Keep mic-only** |
| **C** | Rank/class methods (D/E) beat continuous on top-k with stable accuracy | Prefer **ranking or strength classes** for STK |
| **D** | Normalization audit fails | Reject combined gain (audit says **not D** — formula valid) |
| **E** | Need mass-normalized coupling | Add FOM field or rerun (not available today) |

### Pre-VM expectation

Given v2.1 and v2.2 evidence:

1. **Combined gain is unlikely to beat mic-only on rank** without also improving bridge prediction (bridge MAE 0.278 is the weak link in `bridge × mic`).
2. **Direct combined IDW (B/C)** may reduce error vs post-hoc product (A) if nonlinear interaction is smoother in product space — this is the main hypothesis under test.
3. **Radiation path** remains structurally weak (rank 0.164); `bridge × radiation` is unlikely to exceed mic-only unless radiation matching improves dramatically.

**If no gate is met:** State clearly that STK should remain **mic-only**; combined gain stays experimental.

---

## Recommendation (preliminary)

| Question | Recommendation |
|----------|----------------|
| Production ROM v2.1 | **No change** |
| STK continuous target | **Keep mic-only** until v2.2b VM numbers prove otherwise |
| Combined `bridge × mic` | Valid derived proxy; test with B/C methods on VM |
| Rank / strength classes | Worth reporting; unlikely to replace frequency+mic without top-k ≥ 0.60 |
| More FOM data | Helpful for radiation/back modes; not required to validate combined formula |
| New invariant coupling field | **Defer** — only if E is confirmed after VM run and mass norm is added to FOM |

---

## VM command

From repo root on VM (≥ 30 completed FOM catalogs expected):

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_rom_intensity_v22b_stk_gain.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --holdouts sample_000,sample_005,sample_013,sample_024,sample_027
```

Optional:

```bash
  --k-neighbors 8 \
  --out-dir ROM/classic/experimental_v22b
```

**Outputs (only under experimental path):**

```text
ROM/classic/experimental_v22b/diagnostic_summary.json
ROM/classic/experimental_v22b/per_sample/sample_000.json
ROM/classic/experimental_v22b/per_sample/sample_005.json
ROM/classic/experimental_v22b/per_sample/sample_013.json
ROM/classic/experimental_v22b/per_sample/sample_024.json
ROM/classic/experimental_v22b/per_sample/sample_027.json
```

Does **not** write to `ROM/classic/m4_modal_surrogate.{json,npz}` or `rom_model_manifest.json`.

---

## Implementation files

| File | Role |
|------|------|
| `scripts/v2_b3_m4_rom_stk_gain_targets.py` | Formula audit helper, combined-gain enrichment, strength classes |
| `scripts/audit_rom_intensity_v22b_stk_gain.py` | Five-sample LOO diagnostic CLI |
| `docs/M4_ROM_INTENSITY_V22B_STK_GAIN_DIAGNOSTIC.md` | This report |

---

## Post-run checklist

1. Confirm `completed_catalog_count ≥ 30` in `diagnostic_summary.json`.
2. Verify `training_includes_target: false` and no leakage errors in log.
3. Fill aggregate table from `aggregate_by_method`.
4. Compare method A `bridge × mic` vs A `mic only` — primary STK decision.
5. If B or C beats A on combined targets, note whether mic-only also improved.
6. Update **Decision** section with A–E code and whether improvement gates were met.
