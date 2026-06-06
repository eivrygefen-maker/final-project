# M4 air / mic_output_proxy sanity audit

**Date:** 2026-06-05  
**Urgency:** High — STK/audio ROM depends on `mic_output_proxy`, `radiation_proxy`, `air_share`.  
**Constraint:** Read-only audit. No FEM physics, solver, or production pipeline changes.

---

## Executive verdict

| Code | Meaning | Weight |
|------|---------|--------|
| **A** | Physically expected air/cavity mode family | **Partial** |
| **B** | Harmless duplicate / raw-catalog artifact | **Yes** |
| **C** | Mic proxy artifact (low sensitivity / coarse proxy) | **Yes** |
| **D** | Air mesh/geometry not updating with LHS | **Unlikely in code path; VM verify required** |
| **E** | Checkpoint/cache/reuse bug (cross-sample mode vectors) | **No direct code evidence** |

**Combined assessment:** **A + B + C** — the ~281 Hz and ~390 Hz families are **plausibly real air modes** on a **shared classical template**, but **near-identical `mic_output_proxy` across large LHS geometry swings is not fully explained by physics alone**. Expect **raw-catalog duplication**, **target-driven frequency clustering**, and **proxy insensitivity** to amplify apparent sameness. **Treat peak mic amplitudes as suspicious for STK until deduped + geometry-sensitivity checks pass on VM.**

---

## Observed pattern (user data, samples 000–022)

### ~281.46 Hz family

| sample | f (Hz) | mic_output_proxy | air_share |
|--------|--------|------------------|-----------|
| sample_001 | 281.465591 | 5.7930e-03 | 1.0 |
| sample_002 | 281.468523 | 5.7930e-03 | 1.0 |
| sample_006 | 281.463755 | 5.7930e-03 | 1.0 |
| sample_007 | 281.463726 | 5.7932e-03 | 1.0 |
| sample_011 | 281.472534 | 5.7931e-03 | 1.0 |
| sample_012 | 281.474197 | 5.7931e-03 | 1.0 |
| sample_018 | 281.475417 | 5.7930e-03 | 1.0 |
| sample_019 | 281.477788 | 5.7930e-03 | 1.0 |
| sample_021 | 281.467725 | 5.7930e-03 | 1.0 |

- Frequency span: **~0.014 Hz** across samples  
- `mic_output_proxy` span: **~0.0002e-03** (~0.003% relative)  
- Class: **`air_dominant`**, `dominant_region=air`, `air_share≈1.0`

### ~390.62 Hz family

Repeated peaks near **390.62 Hz**, `mic≈7.599e-03`, `air_share=1.0` (user observation).

---

## Answers to audit questions

### 1. Is air/cavity geometry rebuilt or parameterized per LHS sample?

**Yes — intended per-sample parameterized build.**

| Stage | Behavior |
|-------|----------|
| `sample/sample_input.json` | LHS `geometry.*` parameters per sample |
| `v2_b3_m4_lprod_mesh_build.py` | Reads run-tree `sample_input.json`, calls `build_level_mesh()` with sample geometry |
| `v2_mesh_convergence_mesh.py` | Passes geometry to `build_3d_guitar.py` via `sample_geometry()` |
| `build_3d_guitar.py` | Uses `length`, `width`, `depth`, `hole_radius`, `top_thickness` for cavity, soundhole, air volumes |

Mesh output path: `run_root/lprod/mesh/L_prod/{sample_id}.msh` (per sample).

**Exception:** If geometry fingerprint matches `baseline_coupled_v2`, pipeline may **reuse baseline mesh** (`v2_b3_m4_lprod_interfaces.evaluate_lprod_mesh_checkpoint_readiness`). LHS samples 001+ should **not** match baseline unless parameters coincide.

**VM check:** unique `geometry_fingerprint` per sample (audit script reports this).

---

### 2. Do `hole_radius`, `depth`, `length`, `width` affect air domain / cavity FEM?

**Yes in mesh generation.**

From `build_3d_guitar.py`:

- `inner_depth = D - 2×top_thickness` — cavity height scales with `depth`
- `hole_radius` caps soundhole aperture (facet tag 2)
- `length` / `width` scale body and air channel
- Air volume tag 10 meshed with geometry-dependent size fields

**Caveat:** Air mesh resolution is **coarse** (FOM air element sizes ~4–50 mm). Small LHS changes may produce **nearly identical discrete air eigenfrequencies** on the same template.

---

### 3. Are air DOF / `p_air` masks sample-specific or baseline-reused?

**Sample-specific when checkpoint is built per run.**

| Artifact | Source |
|----------|--------|
| `lprod/checkpoint/region_dof_indices.npz` | Exported from **that sample's** checkpoint mesh (`region_dof_mesh_file` in NPZ) |
| `p_idx_air` | Currently **`p_idx_all`** (all pressure DOFs) — not a geometric cavity subset |
| `u_idx_soundhole` | Traced from **sample mesh** facet tag 2 |

`load_region_dof_bundle()` loads per-checkpoint NPZ. Workers use `run_root/lprod/checkpoint` for that sample (`v2_b3_m4_worker_run_lib`).

**Not baseline-shared** unless checkpoint/mesh were incorrectly copied between samples (no production code path for that on non-matching geometry).

---

### 4. Are air-dominant modes reused/cached between samples?

**No evidence of cross-sample eigenvector cache in code.**

Flow per worker chunk:

1. Load **sample-local** checkpoint operators  
2. SLEPc solve for chunk `target_hz`  
3. `collect_accepted_st_modes()` extracts **in-memory** `x_active` per converged mode  
4. `attach_audio_coupling_to_accepted_mode(x_active=...)` computes scalars from **that solve's vector**

Modes are **not** read from a shared library or prior sample artifact.

**However:** chunk **target frequencies** are planned per sample from scout/L_prod target plan. If plans place targets near **281.46 Hz** for all samples, solvers will report modes **clustered near the same target** — this is **target-driven clustering**, not vector reuse.

---

### 5. Is `mic_output_proxy` from the actual sample mode vector or a fixed template?

**From the actual accepted mode vector when solve path is healthy.**

`v2_b3_mode_audio_coupling.compute_lightweight_audio_coupling()`:

```text
x_full = prolongate_active_to_W(x_active, built)
modal_norm = ‖x_full‖₂
soundhole_rms → mic (priority 1)
cavity pressure → mic (priority 2)
radiation_proxy blend → mic (priority 3)
mic_output_proxy = proxy / modal_norm   (via _safe_norm_ratio)
```

**Fallback (no vector):** `compute_audio_coupling_from_norms()` uses stored `u_norm_W`, `p_norm_W`, `x_norm_W` only — less sample-specific.

**Proxy nature:** `mic_output_proxy` is **not microphone pressure** — it is a **normalized soundhole displacement or cavity pressure scalar** (`mic_output_method` records which).

Near-identical values across samples can occur if:

- Air mode shape on soundhole DOFs scales **proportionally** with `modal_norm`  
- Same `mic_output_method` (e.g. `soundhole_displacement_rms_proxy_v1`)  
- Coarse mesh → similar discrete mode → similar ratio

This is **(C) proxy artifact / low sensitivity**, not necessarily **(E) vector reuse**.

---

### 6. Duplicate raw modes in `modes_catalog.jsonl` — double-counting?

**Duplicates present in raw file. Double-counting depends on consumer.**

| Consumer | Catalog | Double-count? |
|----------|---------|---------------|
| **`modes_catalog.jsonl` file** | **Raw** `all_records` | N/A (stores all chunk accepts) |
| **M4 plots** (`mode_frequency_vs_mic_output_proxy.png`) | **Deduped** (0.05 Hz) | **No** |
| **`modes_summary.json`** | **Deduped** | **No** |
| **ROM Phase-2 train/compare** | **Raw** via `load_fom_modes_catalog()` | **Yes** |
| **STK (future)** | Undefined — must choose | **Risk if raw** |

Aggregation write path (critical):

```python
# v2_b3_m4_aggregate_worker_results.py — writes RAW to jsonl
for rec in sorted(all_records, ...):
    fh.write(json.dumps(rec) + "\n")
# deduped_catalog used for summaries and plots only
```

Docs stating “one record per deduped mode” in `modes_catalog.jsonl` are **incorrect vs current code**.

---

### 7. Aggregation plots / ROM — raw or deduped for mic plots?

| Output | Deduped? |
|--------|----------|
| `mode_frequency_vs_mic_output_proxy.png` | **Yes** (`_try_mode_plots(deduped=...)`) |
| `mode_frequency_vs_radiation_proxy.png` | **Yes** |
| ROM training (`build_m4_rom_from_completed_fom.py`) | **Raw** |
| ROM comparison (`maybe_run_rom_compare`) | **Raw** FOM side |

**Implication:** Inspection of **plots** may show one point per ~281 Hz; inspection of **`modes_catalog.jsonl`** may show **multiple identical rows** — both can be “correct” relative to their source.

---

### 8. Why ~281.46 Hz + ~5.793e-03 + air_share=1.0 across samples?

**Multi-factor explanation (most likely combined):**

1. **(A) Air mode family** — classical coupled template supports a strong air-dominant breathing mode near 280–285 Hz; `air_share=1.0` is consistent.  
2. **(B) Raw duplicates** — multiple chunk targets near 281 Hz → repeated catalog rows with same proxies.  
3. **Target plan** — L_prod chunk targets intentionally cover 60–550 Hz; **~281 Hz is a fixed scout target** → eigenvalues cluster near target (span ~0.01 Hz matches target tolerance culture).  
4. **(C) Proxy insensitivity** — `mic = soundhole_rms / modal_norm` for air modes may vary **much less** than geometry parameters on coarse mesh (0.003% mic spread vs large hole_radius/depth LHS range).  
5. **NOT primarily (E)** — code computes per-solve vectors; no shared mode cache found.

**Suspicious if VM audit shows:**

- Identical `geometry_fingerprint` across samples  
- Identical `region_dof_mesh_file` / `n_nodes` when geometry differs  
- Identical peak mic to >10 decimal places on **deduped** catalog

---

### 9. Why ~390.62 Hz family?

Same mechanisms as §8 — likely **second air/cavity-related family** in 385–395 Hz band on this template. Same `mic≈7.599e-03` clustering suggests **same proxy scaling behavior** and/or **target band overlap**.

Physically plausible as higher air mode; **identical mic to many decimals across diverse geometries** remains **(B)+(C)** concern.

---

### 10. Expected simplified model vs bug?

| Expected from current model | Bug indicator |
|----------------------------|---------------|
| Air peaks exist, often `air_dominant` | All samples share **same** mesh file path |
| Frequencies cluster near L_prod targets | `mic_output` **bitwise identical** on deduped catalog |
| Proxies stable for same template + coarse air mesh | `hole_radius` swing 40–47 mm but peak f **unchanged** at 6+ decimals |
| Raw catalog > deduped count | Worker reads **wrong sample** checkpoint |

**Current evidence fits “expected coarse proxy + raw duplicates + target clustering” more than “geometry not updating” — but VM fingerprint/mesh audit is mandatory before STK trust.**

---

## Required VM audit command

Read-only script (added in this audit):

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/audit_air_mic_proxy_sanity.py \
  --lhs-json ROM/classic/lhs_pool.json \
  --samples 0-22 \
  --csv-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_air_mic_proxy_sanity_data.csv \
  --json-out FEM/experiments/active_domain_validation/physics_integrity/docs/M4_air_mic_proxy_sanity_data.json
```

### Report columns (per sample)

```text
sample_id
hole_radius, depth, length, width
est_cavity_volume_m3          # rough L×W×(D−2t); not CAD-exact
geometry_fingerprint
raw_mode_count / deduped_mode_count
exact_duplicate_groups
peak_281_freq / peak_281_mic / peak_281_air_share / peak_281_coupling_class
peak_281_chunk_id / peak_281_mic_method
peak_390_freq / peak_390_mic
ckpt_region_dof_mesh_file     # must differ per sample if geometry differs
ckpt_mesh_n_nodes
```

### Pass / fail heuristics (sanity, not CI gate)

| Check | Pass | Fail → investigate |
|-------|------|-------------------|
| `unique_geometry_fingerprints` | ≈ number of samples | **(D)** geometry not in sample_input |
| `ckpt_region_dof_mesh_file` | Unique per differing geometry | **(D)/(E)** mesh/checkpoint reuse |
| `freq_span_hz` (281 cluster) | >0.05 Hz OR correlates with cavity volume | **(C)** target locking only |
| `mic_rel_span` (281 cluster) | >1% OR correlates with hole_radius | **(C)** proxy useless for STK |
| `deduped peak_281` vs raw duplicates | 1 mode per band on deduped | **(B)** raw inspection misleading |
| `all_mic_identical_10dp` | false on deduped | **(E)/(C)** copy or quantization |

---

## Task B table template (fill from VM script)

Run audit script, then paste top deduped air/mic modes for 018–022:

| sample | f (Hz) | mic | radiation | class | dom | top/back/air | chunk | deduped? |
|--------|--------|-----|-----------|-------|-----|--------------|-------|----------|
| 018 | | | | | | | | |
| 019 | | | | | | | | |
| 020 | | | | | | | | |
| 021 | | | | | | | | |
| 022 | | | | | | | | |

**Pre-filled expectation from user data (raw catalog peaks):** all show `air_dominant`, `air_share=1.0`, f≈281.47±0.01, mic≈5.793e-03.

---

## Impact on ROM / STK

| Area | Risk |
|------|------|
| ROM intensity training | Raw duplicates + clustered targets → **misleading k-NN** for `mic_output_proxy` |
| ROM metrics | Relative error inflated; may measure **target-cluster noise** not geometry learning |
| STK playback | If peak air modes are **proxy-insensitive**, ROM cannot learn audible differences from FOM |
| Frequency ROM | **Unaffected** — different code path, validated ~1.2% |

**Immediate ROM-side mitigation (no FOM change):** Intensity ROM v2.1 — train/compare on **deduped** catalog, log/norm targets (see [M4_ROM_INTENSITY_PREDICTION_AUDIT.md](M4_ROM_INTENSITY_PREDICTION_AUDIT.md)).

**FOM-side follow-up (future, not this task):** Consider writing **deduped** rows to `modes_catalog.jsonl` OR adding `modes_catalog_deduped.jsonl` without changing solver physics.

---

## Recommended next steps (ordered)

1. **Run `audit_air_mic_proxy_sanity.py` on VM** for samples 000–022; attach CSV/JSON to this doc folder.  
2. **Compare `geometry_fingerprint` and `ckpt_region_dof_mesh_file`** — confirm **(D)** ruled out.  
3. **Inspect deduped 281/390 peaks** — if still identical mic → **(C)** dominant; defer STK peak trust.  
4. **Correlate** `peak_281_freq` vs `est_cavity_volume_m3` and `hole_radius` (scatter). Weak correlation + identical mic → proxy/mesh limited.  
5. **ROM:** implement deduped train/compare + log/norm intensity metrics (v2.1).  
6. **Optional FOM doc fix:** clarify raw vs deduped in `B3_M4_MODE_AUDIO_COUPLING_METADATA.md` (documentation only).

---

## Code references

| Topic | File |
|-------|------|
| Mesh per sample | `v2_b3_m4_lprod_mesh_build.py`, `build_3d_guitar.py` |
| Geometry LHS | `v2_b3_m4_lprod_interfaces.extract_geometry_dict` |
| Mic proxy compute | `v2_b3_mode_audio_coupling.py` |
| Solve attach | `v2_b3_st_sinvert_solver_lib.collect_accepted_st_modes` |
| Raw jsonl write | `v2_b3_m4_aggregate_worker_results.py` (~line 455) |
| Deduped plots | `v2_b3_m4_aggregate_worker_results._try_mode_plots` |
| ROM raw load | `v2_b3_m4_rom_fom_compare_lib.load_fom_modes_catalog` |
| Region DOF export | `v2_b3_synthesis_export.export_region_dof_indices_npz` |
| Audit script | `audit_air_mic_proxy_sanity.py` |

---

## Acceptance criteria (report)

| # | Question | Answer |
|---|----------|--------|
| 1 | ~281 Hz physically plausible? | **Partially yes (A)** — air family expected; **identical mic across LHS is not** |
| 2 | Duplicates affect training/plots? | **Plots: no (deduped). ROM: yes (raw).** |
| 3 | ROM intensity raw/log/norm? | **Raw absolute IDW** today |
| 4 | Minimal improvement? | **Deduped ROM + log/norm intensity (v2.1)**; VM audit first |
| 5 | Validation command? | `audit_air_mic_proxy_sanity.py --samples 0-22` + holdout ROM compare loop |
