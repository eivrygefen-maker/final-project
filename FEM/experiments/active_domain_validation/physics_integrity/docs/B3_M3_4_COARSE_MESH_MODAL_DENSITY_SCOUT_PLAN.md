# B3 M3.4 — Coarse-mesh modal-density scout (planning)

**Status:** Planning and inspection only — no mesh generation, no Stage A/B execution, no cleanup.  
**Supersedes (strategy):** Wide-band discovery on the **full `L_prod` checkpoint** as the primary modal-density scout. See [`B3_M3_4_PRE_COARSE_FREQUENCY_PLANNER.md`](B3_M3_4_PRE_COARSE_FREQUENCY_PLANNER.md) for frequency-grid / discovery-flag mechanics **after** a scout checkpoint exists.  
**Tool:** `scripts/v2_b3_coarse_mesh_scout_plan.py` (dry-run report, `will_execute=false`)  
**Wrong-direction evidence (discard):** `target_density_discovery_60_550_step15_m3exec2` on `L_prod` — expensive mesh + env/setup failure; not zone calibration input.

---

## 0. Correct M3.4 concept

```text
coarse FEM mesh (FOM geometry, coarser characteristic lengths)
  → broad modal discovery ~60–550 Hz
  → approximate mode-frequency distribution
  → modal-density windows
  → ZONE 1/2/3 guidance
  → design smarter L_prod target sets (spacing / half-width)
```

**“Coarse grid” means coarse FEM mesh**, not coarse frequency spacing on `L_prod`.

The scout mesh is a **planning instrument**, not final physics. Zones inform `L_prod` targeting; they are **not** promoted as production validation.

---

## 1. Verified `L_prod` mesh sizing (authoritative sources)

Values below come from `configs/v2_mesh_convergence_manifest.json` (`l_prod_source.controls`) and `FEM/geometry/build_3d_guitar.py` when `FEM_ALLOW_FOM=1` and `FEM_MESH_LC_SCALE=1.0`.

| Region | Manifest field | Code / manifest value | mm |
|--------|----------------|----------------------|-----|
| Wood shell (top/back/ribs surfaces) | `wood_surface_size_m` | FOM default `0.007` | **7** |
| Plate / through-thickness curves | `wood_thickness_size_m` | FOM default `0.001` | **1** |
| Air (graded field min) | `air_threshold_size_min_m` | FOM `0.004` (non-FOM preview uses `0.003`) | **4** |
| Air (graded field max) | `air_threshold_size_max_m` | `0.050` | **50** |
| Air distance band | `air_threshold_dist_min/max_m` | `0.015` / `0.25` | 15 / 250 |

**Order-of-magnitude operator notes (not separate manifest fields):**

- Operator summaries sometimes cite “~5 mm wood / ~7 mm air” as **effective** resolution after Gmsh threshold fields and geometry — treat manifest/code numbers as **control targets**, not post-mesh element-size histograms.
- Soundhole band uses `wood_thickness_size` as hole LC target (`build_3d_guitar.py`).

**M3-validated `L_prod` checkpoint reference:** `lhs_pilot_001_timing_m3exec2` — **active_dim ≈ 316,017** (orchestrator timing PASS); mesh path in overlays:  
`v2_mesh_convergence/mesh/L_prod/baseline_coupled_v2.msh`.

---

## 2. Where mesh sizes are defined

| Layer | Role |
|-------|------|
| **`configs/v2_mesh_convergence_manifest.json`** | Documents FOM vs validation base controls; per-level `lc_scale`, `build_env`, labels |
| **`FEM/geometry/build_3d_guitar.py`** | Actual Gmsh characteristic lengths; profiles: `FEM_VALIDATION_MESH=1` vs `FEM_ALLOW_FOM=1`; uniform scale via `FEM_MESH_LC_SCALE` |
| **`scripts/v2_mesh_convergence_mesh.py`** | Builds per-level `.msh` + audit; writes `effective_controls_m` = base × `lc_scale` |
| **`FEM/configs/guitar_3d.json`** | Geometry/materials/solver **mesh path** — **no** per-region LC fields |
| **Per-level build configs** | `v2_mesh_convergence/configs/v2_mesh_convergence_build/<level>_<case>.json` (generated at build) |
| **Stage A** | `v2_b3_checkpoint_export.py` — uses mesh from `resolved_core_config.json` → `solver.mesh_file` |

Mesh generation CLI pattern:

```bash
# Generic (all manifest levels for a case):
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mesh_convergence.py ...

# B3 dev smoke (validation pipeline only):
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_B3_dev_coarse_mesh_build.py
```

---

## 3. Mesh levels currently in the manifest

| Level | Pipeline | `lc_scale` | Notes |
|-------|----------|------------|-------|
| `L0` | `FEM_VALIDATION_MESH=1` | 1.0 | Validation baseline |
| `L_mid` | validation | ~0.707 | Finer validation |
| `L_dev_coarse` | validation | **2.0** | Solver smoke only; **not** FOM geometry |
| `L_dev_refined` | validation | 1.6 | ~25k–45k active (target) |
| `L_dev_dense` | validation | 1.2 | Observed ~41.5k active (manifest note) |
| `L_prod` | `FEM_ALLOW_FOM=1` | 1.0 | Production FOM |
| `L_check` | FOM | ~0.707 | Optional finer-than-prod |

**Stage A `ALLOWED_MESH_LEVELS` today:** `L_mid`, `L_dev_dense`, `L_prod` only (`v2_b3_checkpoint_export.py`).

---

## 4. Existing coarse / scout checkpoints?

| Artifact | Expected on disk | This workspace |
|----------|------------------|----------------|
| `mesh/L_prod/baseline_coupled_v2.msh` | After prod mesh build | **Not present** (no `.msh` under `v2_mesh_convergence/`) |
| `mesh/L_dev_coarse/...` | After dev coarse build | **Not present** |
| `diagnostics/st_worker_scaling_L_prod_lhs_pilot_001_timing_m3exec2` | M3 PASS checkpoint | **Not present** (VM artifact; referenced in specs) |
| Any `L_scout_coarse` checkpoint | N/A — level **not defined yet** | **No** |

**Conclusion:** No local scout checkpoint; M3 `L_prod` checkpoint exists only on the execution VM unless copied back.

---

## 5. Stage A command to build a coarse scout checkpoint (preview only)

**Prerequisites (not done in this planning step):**

1. Add manifest level **`L_scout_coarse`** (see §7–8).
2. Build mesh: `v2_mesh_convergence/mesh/L_scout_coarse/baseline_coupled_v2.msh`.
3. Extend Stage A `--mesh-level` choices to include `L_scout_coarse` (or map scout mesh via `--core-config` with explicit `solver.mesh_file` while keeping export contract).

**Dry-run Stage A preview** (after mesh exists; production venv; timing overlay example):

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/v2_b3_checkpoint_export.py \
  --mesh-level L_scout_coarse \
  --B3-block-compose-backend csr_bulk \
  --B3-synthesis-region-dofs off \
  --core-config "FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/config_overlays/lhs_pilot_001_timing/resolved_core_config.json" \
  --output-dir "FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/st_worker_scaling_L_scout_coarse_scout_m1"
```

**Mesh-build preview** (new level — pattern mirrors `run_v2_B3_dev_coarse_mesh_build.py` but FOM env):

```bash
python FEM/experiments/active_domain_validation/physics_integrity/scripts/run_v2_mesh_convergence.py \
  --levels L_scout_coarse --cases baseline_coupled_v2
```

*(Requires `L_scout_coarse` in manifest and, for non-uniform sizing, small `build_level_mesh` / `build_3d_guitar.py` extension — see §8.)*

---

## 6. Mesh level naming

| Candidate | Verdict |
|-----------|---------|
| **`L_scout_coarse`** | **Recommended** — FOM geometry, explicit scout controls, not confused with validation `L_dev_*` smoke meshes |
| Reuse **`L_dev_coarse`** | **Reject** — validation pipeline (`FEM_VALIDATION_MESH`); at `lc_scale=2.0` effective sizes are ~**28 mm** wood / **6 mm** plate / **18 mm** air min — far from scout targets and wrong CAD profile |

---

## 7. Mapping proposed scout sizing (~3 / ~8.5 / ~11 mm)

**Target (operator compromise, mm):**

| Region | `L_prod` (verified) | Scout target |
|--------|---------------------|--------------|
| Plate / thickness | 1 | **~3** |
| Wood shell | 7 | **~8–9** |
| Air (min) | 4 | **~10–12** |

**Uniform `FEM_MESH_LC_SCALE` on FOM cannot hit all three simultaneously:**

| `lc_scale` | Plate mm | Wood mm | Air min mm |
|------------|----------|---------|------------|
| 1.0 (`L_prod`) | 1.0 | 7.0 | 4.0 |
| 1.21 | 1.2 | 8.5 | 4.8 |
| 1.5 | 1.5 | 10.5 | 6.0 |
| 3.0 | 3.0 | 21.0 | 12.0 |

**Recommended manifest entry (explicit controls, `lc_scale=1.0`):**

```json
"L_scout_coarse": {
  "label": "FOM coarse modal-density scout (planning only; not final validation)",
  "build_env": {"FEM_ALLOW_FOM": "1"},
  "lc_scale": 1.0,
  "explicit_controls_m": {
    "wood_thickness_size_m": 0.003,
    "wood_surface_size_m": 0.0085,
    "air_threshold_size_min_m": 0.011,
    "air_threshold_size_max_m": 0.055
  },
  "run_gates_on_build": true,
  "solver_smoke_test_only": false,
  "modal_density_scout_only": true,
  "not_authorized_for_final_physics_validation": true
}
```

**Implementation note:** `explicit_controls_m` is **not** wired today; `build_3d_guitar.py` only reads env profile + `FEM_MESH_LC_SCALE`. Safest follow-up: teach `v2_mesh_convergence_mesh.py` to pass optional per-field env overrides (or a single `FEM_MESH_EXPLICIT_CONTROLS_JSON`). Until then, **`lc_scale≈1.2–1.5`** is a fallback compromise (closer on wood/air, plate still finer than 3 mm unless scale ≥ 3).

---

## 8. Expected active dimension vs `L_prod`

| Level | Basis | Active dim (estimate) |
|-------|--------|------------------------|
| `L_prod` | M3 m3exec2 measured | **~316,017** |
| `L_dev_dense` | Manifest observation (validation mesh) | **~41,501** |
| `L_scout_coarse` (proposed explicit controls) | ~1.4–2.0× coarser than prod in mean LC → ~(1.6³–2.0³) fewer DOFs | **~25,000–80,000** (wide band; **measure after first build**) |

Scout should be **much cheaper** than `L_prod` but **finer** than validation `L_dev_coarse` (which uses a different mesh profile entirely).

---

## 9. Broad frequency range for scout

| Band | Hz | Notes |
|------|-----|-------|
| **Scout discovery** | **60–550** | Primary planning band |
| **Validated `full9` slice** | 221.5–264.0 | `L_prod` timing reference only |
| **Legacy acceptance (default)** | 220–265 | Without `--B3-discovery-mode` |

Use Gate A discovery flags for 60–550 on the **scout checkpoint**, not on `L_prod` as the main density instrument.

---

## 10. Solver path for scout

```text
(approved) Build L_scout_coarse .msh
  → Stage A: v2_b3_checkpoint_export.py (production venv)
  → Stage B: v2_b3_checkpoint_target_density_experiment.py
        --B3-discovery-mode
        --discovery-band-hz 60 550
        --target-window-half-width-hz 7.5   # spacing 15 Hz → half-width = step/2
        --start-hz 60 --stop-hz 550 --spacings-hz 15
  → Post-process: deduped accepted frequencies, counts per window, density metrics
  → Zone proposal (relative, not hardcoded thresholds yet)
```

**Not** the primary path: `target_density_discovery_60_550_step15_m3exec2` on `st_worker_scaling_L_prod_*`.

Stage C / rich modal: **not required** for zone scouting.

---

## 11. Required scout outputs

| Output | Purpose |
|--------|---------|
| `unique_accepted_frequencies_hz` (deduped) | Mode list for density analysis |
| Per-target `accepted_frequencies_hz` | Windowed discovery trace |
| Mode count per frequency window (e.g. 15 Hz bins) | Density curve |
| Modal density = modes / Hz per window | Zone inference input |
| Proposed **ZONE 1 / 2 / 3** map | Dense → tighter `L_prod` targets; sparse → wider spacing |
| Run manifest + `result.json` policy fields | Discovery band + half-width audit trail |

Planner artifact schema can reuse `b3_coarse_frequency_plan_v2` **after** checkpoint path points at scout mesh, not `L_prod`.

---

## 12. Zone inference (first pass)

1. Bin accepted modes into fixed windows (e.g. 15 Hz) across 60–550 Hz.
2. Compute **relative** density per bin vs median density (or vs full-band mean).
3. Rank contiguous bands into tertiles or manual review:
   - **ZONE 1 (dense):** high relative count → smaller `L_prod` target spacing, narrower discovery half-width.
   - **ZONE 2 (moderate):** mid density → default spacing (e.g. 10–15 Hz).
   - **ZONE 3 (sparse):** low density → wider spacing; fewer targets.
4. **Do not** hardcode Hz thresholds until after first scout run; calibrate against `full9` overlap as sanity check only.

---

## 13. Known risks

| Risk | Mitigation |
|------|------------|
| Frequency shift coarse → `L_prod` | Zones are **guidance**; validate dense zones with `full9`-class runs |
| Missing local modes on coarse mesh | Do not treat scout as completeness proof |
| Mode ordering / branch swaps | Use density **counts**, not single-mode tracking |
| Air–structure coupling distortion | Compare zone **shape**, not absolute Hz edges |
| Non-uniform LC mapping error | Verify `effective_controls_m` in mesh audit after build |
| Wrong mesh profile (`L_dev_*`) | Always scout on **FOM** (`FEM_ALLOW_FOM`) |

---

## 14. Safest first non-execution step (this milestone)

1. Run `v2_b3_coarse_mesh_scout_plan.py` (dry-run JSON/MD report).
2. Confirm manifest/code sizing table vs operator targets (§7).
3. Review prerequisite code changes (`L_scout_coarse`, explicit controls, Stage A mesh-level allowlist).
4. Approve mesh build + Stage A on VM.
5. Only then: Stage B discovery on scout checkpoint.

**Explicitly out of scope until approved:** Gmsh build, Stage A export, Stage B solves, deleting `m3exec*` or wrong-direction run dirs.

---

## 15. Workflow summary

```text
1. Inspect mesh levels + sizing          ← this document + scout plan script
2. Define L_scout_coarse (+ code hook)  ← manifest / build / export allowlist
3. Dry-run Stage A command preview      ← scout plan script
4. [approved] Build coarse .msh + Stage A checkpoint
5. [approved] Stage B discovery 60–550 on scout checkpoint
6. Post-process densities → ZONE 1/2/3
7. Plan expensive L_prod target sets    ← frequency planner on L_prod, zone-informed
```

---

## 16. Cross-references

- [`B3_M3_4_GATE_A_ACCEPTANCE_DISCOVERY_MODE.md`](B3_M3_4_GATE_A_ACCEPTANCE_DISCOVERY_MODE.md) — discovery flags for Stage B wide band  
- [`B3_M3_4_PRE_COARSE_FREQUENCY_PLANNER.md`](B3_M3_4_PRE_COARSE_FREQUENCY_PLANNER.md) — target grid / half-width math (apply **after** scout zones)  
- [`B3_M3_ORCHESTRATOR_CONTRACT.md`](B3_M3_ORCHESTRATOR_CONTRACT.md) — `L_prod` timing orchestration (unchanged)  
- `configs/v2_mesh_convergence_manifest.json` — mesh level source of truth
