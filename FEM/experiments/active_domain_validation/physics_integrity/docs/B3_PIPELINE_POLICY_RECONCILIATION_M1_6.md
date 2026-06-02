# B3 pipeline policy reconciliation (M1.6)

## 1) Context and validated reference snapshot

This document reconciles legacy solver/worker/mode-selection policies with the validated official A+B+C checkpoint/rich pipeline.

Validated reference:

- Stage A checkpoint PASS: `v2_mesh_convergence/diagnostics/st_worker_scaling_L_prod_rich_safe_20260601T164739Z`
- Stage B rich PASS: `v2_mesh_convergence/diagnostics/solver_benchmarks/checkpoint_solve_mkl_pardiso_full9_20260601T203438Z`
- Stage C PASS: `.../checkpoint_solve_mkl_pardiso_full9_20260601T203438Z/rich_modal_post/`
- `mode_count = 115`
- `schema = b3_rich_modal_post_v1`
- Official archive: `~/final-project-archives/archive_official_A_B_C_rich_PASS_20260601T203438Z.tar.gz`

Official migration governance and run-manifest references:

- `docs/B3_MIGRATION_TO_OFFICIAL_PIPELINE_M0.md`
- `docs/B3_M1_PIPELINE_RUN_MANIFEST_SPEC.md`

---

## 2) Reconciliation tables

### A. Brain / worker architecture

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| Monolithic master-worker sweep controller | `FEM/scripts/fem_master_dynamic.py`, `FEM/scripts/fem_worker_single.py` | Parallel spectral sweep, candidate merge, ROM packaging feed | Yes (controller + workers concept) | No (legacy sweep semantics tied to old candidate flow) | M2 orchestrator over Stage A/B/C manifests | keep concept but redesign implementation | Competing policy stacks and hidden behavior drift | Lose existing fallback operational path before replacement | likely | partial |
| Worker process does solve + local filtering + temp vector writes | `fem_worker_single.py` | Keep worker output volume manageable and merge-friendly | Partially | Partially (for legacy only) | Stage B checkpoint solve + rich catalog export | replace | Early irreversible filtering incompatible with new full catalog goals | Burst output growth in old path if abruptly removed | likely | no |
| Central “brain” handles retries, schedule, and merge policy in one script | `fem_master_dynamic.py` | End-to-end old pipeline control | Yes | No | Split policy: Stage scripts + manifest helper + future orchestrator | keep concept but redesign implementation | Hard to reason/audit when mixed with new A/B/C | Temporary loss of automation convenience | decided | partial |
| Environment handling embedded per legacy command path | `run_pipeline.py`, `fem_master_dynamic.py` | Run old A/B/C-style ROM pipeline from one command | Yes | No | Explicit stage env contract (`production .venv` vs `solver-mkl`) in manifests | replace | Wrong-env failures hidden until runtime | Extra manual steps until orchestrator exists | decided | yes |

### B. Per-worker mode request limits

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| Fixed/derived per-worker mode caps (e.g. ~40/70/100 style caps) | `fem_worker_single.py`, `fem_master_dynamic.py`, `FEM/configs/guitar_3d.json` | Bound memory/solve time and reduce over-harvest | Yes (need bounded budgets) | No (numbers tied to old sweep mechanics) | Stage B `nev/ncv` + future target-planner policy | keep concept but redesign implementation | Carrying obsolete caps may miss useful modes or bias catalogs | Unbounded solves can hurt throughput | likely | partial |
| Global `num_modes` defaults in old config | `FEM/configs/guitar_3d.json` | One knob for old broad sweep path | Yes | No (for new official path) | Stage B run policy per mode (`timing`/`rich`/`synthesis`) | replace | Mismatch between GUI config and checkpoint pipeline outputs | Loss of quick old tuning knob | decided | no |
| Harvest caps and rigid buffers in old solver profile | `fem_main_3d.py`, `guitar_3d.json` | Stabilize old SLEPc path under broad ranges | Partially | Partially (legacy fallback only) | Stage B acceptance + explicit summary metadata | keep concept but redesign implementation | Over-constrains new checkpoint solves | Removes some stability guardrails in legacy mode | needs review | no |

### C. Candidate harvesting and filtering

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| Pressure/wood thresholds and classifier-based candidate gate | `fem_harvest_filter.py`, `fem_master_dynamic.py`, `fem_worker_single.py` | Reject weak/spurious candidates early | Yes | No (legacy coupled to old candidate log model) | Stage B accepted-mode contract (`v2_b3_st_sinvert_solver_lib.py`) + Stage C synthesis filters | keep concept but redesign implementation | Double filtering and unexplained drops vs rich catalog | Temporary increase in catalog noise | likely | no |
| Acoustic-only / u-only rejection rules | legacy solver config + harvest filter | Avoid decoupled artifacts in old ROM pool | Yes | Partially | Stage B acceptance already includes support and leakage checks | replace | Can remove physically valid edge cases or bias study outputs | More decoupled artifacts in transitional runs | decided | no |
| Relative residual/error acceptance | old and new SLEPc paths | Numerical quality gate | Yes | Yes (concept), not all thresholds | Stage B `eps_compute_error_relative` acceptance and exported metadata | keep concept but redesign implementation | Inconsistent thresholds across paths | Lower quality if no quality gate | decided | no |
| Weak-coupling / sigma-spurious reject toggles | `guitar_3d.json`, `fem_main_3d.py` | Filter target-locked spurious modes in old path | Partially | Partially (legacy diagnostics) | Stage B accepted-mode criteria and catalog diagnostics | deprecate (legacy) | Obsolete toggles mislead operators | Lose investigative knobs | likely | no |

### D. Mode selection and dedupe

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| Fixed “select ~100 final modes” style behavior | old SORTING/merge flow | Build compact ROM set | Yes (subset selection needed) | No (fixed count not portable) | Full rich catalog + policy-driven subset for ROM/STK | keep concept but redesign implementation | Arbitrary fixed count hides important bands | Large uncurated datasets if no subset policy | likely | partial |
| Frequency proximity dedupe | old merge and current Stage C report | Avoid duplicate modes across nearby shifts | Yes | Yes (as baseline) | Stage C `frequency_dedupe` report (non-destructive) | keep as-is (baseline), add future MAC dedupe | Over-dedup can discard distinct shapes | Too many duplicates if no dedupe | decided | no |
| MAC-based dedupe | planned in docs | Shape-aware dedupe | Yes | Not yet implemented | Future Stage C/selection phase | unknown / needs experiment | Premature MAC policy may drop useful variants | Delay adds manual review burden | blocked by experiment | no |
| Separation of full catalog vs synthesis subset | implicit old behavior | Keep auditability and practical runtime set | Yes | No (old mixed concerns) | `rich_modal` full catalog + selected synthesis/ROM/STK subset policy | replace | Loss of traceability if only subset stored | Storage growth if subset not introduced | decided | partial |

### E. Target strategy

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| Uniform target spacing sweeps | legacy dynamic scheduler + experiments | Coverage across broad bands | Yes | Partially | Current `full9` in Stage B + future target planner/zones | keep concept but redesign implementation | Excess solve cost for low-value bands | Coverage holes if removed without planner | likely | partial |
| Target density/alignment experiments | `v2_b3_checkpoint_target_density_experiment.py`, `...target_alignment...` and outputs | Validate spacing/anchor strategies | Yes (knowledge) | Outputs historical; scripts still useful for policy design | Use as reference inputs to target planner | keep as legacy/historical | Mistaking historical outputs for active policy | Lose evidence base for planner decisions | decided | no |
| `full9` target set | Stage B checkpoint solve | Stable validated baseline set | Yes | Yes | Current Stage B default | keep as-is (for initial LHS baseline) | Undercoverage if used as universal policy | Loss of validated anchor if changed too early | decided | no |
| Dense/sparse zone adaptation | old scheduler zones | Dynamic compute allocation | Yes | No (old implementation tied to legacy merge metrics) | Future planner policy in M2+ | keep concept but redesign implementation | Importing old thresholds may misfit checkpoint pipeline | Less efficient early LHS without zones | likely | no |

### F. Rich export policy

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| Rich modal data always-on assumptions | legacy expectations in old flows | Convenience for downstream reuse | No | No | Opt-in Stage B rich (`--B3-export-rich-modal-data`) | replace | Storage explosion and benchmark distortion | Fewer rich-ready runs until selection policy matures | decided | no |
| Mixed expectations on “microphone pressure” outputs | legacy naming assumptions | Audio-facing interpretation | Yes (audio outputs needed) | No (naming/semantics obsolete) | Stage C audio/radiation proxy wording + explicit non-microphone flag | replace | Misinterpretation of proxy as physical microphone pressure | Confusion for legacy consumers | decided | no |
| LHS rich policy | not codified historically | Balance compute/storage vs downstream needs | Yes | No | M0/M1 policy: all A+B, subset rich, subset C | keep concept but redesign implementation | Rich everywhere becomes infeasible | Insufficient rich samples for ROM/STK | decided | partial |

### G. Stage C / synthesis policy

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| Stage C run on every solve | implicit expectation | Immediate synthesis availability | No | No | Stage C only for synthesis/ROM/STK subset | replace | Unnecessary cost and artifact bloat | Delayed synthesis readiness for unselected points | decided | no |
| Structural region participation when region indices missing | historical zeros sometimes interpreted physically | Produce simple outputs always | Yes (must report availability) | No (zero-as-physical is invalid) | `unavailable_region_indices` + null structural values + warnings | keep as-is | False physical conclusions from zero values | Reduced structural insight until indices available | decided | no |
| Pressure/cavity proxy computation without structural regions | current Stage C | Preserve useful acoustic proxy even when structural regions unavailable | Yes | Yes | Current Stage C behavior | keep as-is | Could be overtrusted without structural context | Lose valuable proxy signal | decided | no |
| Region DOF best_effort | Stage A/Stage C optional path | Improve structural regional metrics | Yes | Partially (environment/segfault risk bounded by subprocess) | Optional best_effort subprocess | keep concept but redesign implementation | Forcing it can destabilize runs | Lower fidelity structural metrics if never used | likely | no |

### H. GUI and old app integration

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| GUI triggers old full FEM / ROM pipeline directly | `gui/app.py`, `FEM/scripts/run_pipeline.py` | Interactive full workflow from UI | Yes | No (for new official path) | Future GUI bridge to Stage A/B/C manifests | keep concept but redesign implementation | Users unknowingly bypass official A+B+C path | Lose operational UI while replacement not ready | needs review | no |
| GUI expects old SORTING/ROM artifacts | `gui/app.py`, `FEM/SORTING/*` assumptions | Preview and immediate usage of old outputs | Partially | No | New adapters consuming Stage B/C outputs | replace | Incompatibility with new output contracts | UI regression if old assumptions removed before adapter exists | needs review | no |
| Expose rich/C controls in GUI | not currently standardized | Operator control over expensive paths | Yes | Not yet | Planned UI controls for timing/rich/C modes | unknown / needs experiment | UI complexity and misuse risk | Missed operator visibility if omitted | needs review | no |

### I. Geometry / mesh workflow

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| Build mesh from `guitar_3d.json` app path | `FEM/geometry/build_3d_guitar.py`, GUI flow | Interactive geometry and mesh generation | Yes | Partially | Stage A reproduction uses `v2_mesh_convergence_manifest.json` + `mesh/L_prod` | keep and integrate | Confusing dual source-of-truth for mesh inputs | Lose existing mesh generation path | likely | partial |
| `v2_mesh_convergence_manifest.json` as reproducibility contract | `physics_integrity/configs/v2_mesh_convergence_manifest.json` | Define mesh levels/cases and L_prod source | Yes | Yes | Stage A required input | keep as-is | Stage A irreproducibility if drifted/removed | None; it is required | decided | yes |
| `mesh/L_prod` as Stage A prerequisite | `v2_mesh_convergence/mesh/L_prod/` | Concrete L_prod mesh artifacts for export runs | Yes | Yes | Stage A required input | keep as-is | Export failures / non-reproducibility | None; it is required | decided | yes |
| Canonical `coupled_physical_core_v2.json` | `physics_integrity/configs/coupled_physical_core_v2.json` | Frozen baseline and synthesis scalar metadata source | Yes | Yes (minimal canonical form) | Stage A synthesis metadata source (`pressure_dof_scale`, `fsi_coupling_gain`) | keep as-is | Metadata drift if overwritten by legacy dumps | Loss of frozen baseline contract | decided | partial |

### J. Cleanup/deprecation consequences (policy-level only)

| Old mechanism / policy | Where it lives | Original purpose | Concept still relevant? | Specific old implementation still valid? | New pipeline equivalent | Recommended action | Risk if kept blindly | Risk if removed too early | Decision status | Blocks first LHS? |
|---|---|---|---|---|---|---|---|---|---|---|
| Keep all legacy outputs live indefinitely | diagnostics trees | Historical forensic value | No | No | P0/P1 keep + archive-first for others | deprecate | Storage sprawl obscures official path | Loss of forensic history if unarchived | decided | no |
| Delete without policy linkage | ad-hoc cleanup | Reclaim disk quickly | No | No | Dry-run classification + migration-driven archive policy | replace | Accidental loss of needed references | Disk pressure persists temporarily | decided | no |
| Track runtime manifests in git by default | `pipeline_runs/manifests`, `index` | Ad-hoc provenance | No | No | Runtime ignored; docs/specs tracked | replace | Repo noise/churn and accidental coupling | Reduced visibility if no tracked docs | decided | no |
| Keep policy docs/specs out of source control | migration docs area | Informal operations | No | No | Track M0/M1/M1.6 specs in docs | replace | Policy drift and knowledge loss | Slight doc maintenance overhead | decided | no |

---

## 3) Special attention examples (explicit)

1. **Old worker asks fixed number of modes around target**  
   Decision: keep concept (bounded resources), redesign implementation in M2 planner; do not carry old numbers as default.

2. **Old system selects fixed number of final modes from candidates**  
   Decision: replace with policy-driven subset selection from full rich catalog; avoid hardcoded fixed-count defaults.

3. **Old `sifter_*` / `harvest_*` config fields**  
   Decision: legacy/diagnostic. Do not use as default official policy for new A+B+C/LHS path.

4. **Old target-density/alignment experiment outputs**  
   Decision: historical evidence for planner design; not current operational policy.

5. **Old monolithic worker scaling path**  
   Decision: not default; keep as legacy fallback until M2 wrapper/orchestrator is validated.

6. **`fem_master_dynamic.py` / `fem_worker_single.py` conceptual role**  
   Decision: controller/worker concept still relevant; concrete implementation should not be reused wholesale for A+B+C policy.

7. **`fem_main_3d.py` role**  
   Decision: remains Stage A operator-build dependency now; later can be wrapped/refactored, not replaced in M1.6.

8. **Policy for first LHS**  
   Decision: all points Stage A+B timing/scalars; selected subset Stage B rich; synthesis subset Stage C.

---

## 4) Policy lists

### 4.1 Already decided

- A+B+C checkpoint/rich pipeline is canonical operational path.
- Rich export is opt-in only.
- Stage C runs only for synthesis subset.
- Stage C structural null-vs-zero policy with `unavailable_region_indices` is mandatory behavior.
- Runtime manifests/index are runtime data; docs/specs are tracked.
- Cleanup is migration-following, not migration-driving.

### 4.2 Must be designed before first LHS

- M2 orchestrator contract that executes A/B with manifest status transitions.
- Rich/synthesis subset selection rule (v0 policy, deterministic).
- Stage-level failure/retry and environment switching policy.
- Initial target-planner policy beyond baseline `full9` where needed.

### 4.3 Can wait until after first LHS

- MAC-based dedupe implementation.
- GUI integration for timing/rich/C controls and browsing.
- Advanced zone-planner heuristics from legacy density metrics.
- Deep legacy cleanup/deprecation execution.

---

## 5) One recommended next step

Create `docs/B3_M2_LHS_INTEGRATION_PLAN.md` (planning-only) that defines:

- first pilot batch size,
- mandatory A/B manifest transitions,
- v0 rich/synthesis selection rule,
- explicit env handoff (`production .venv` / `solver-mkl`) and retry semantics.

No code/refactor/cleanup actions should occur before that M2 plan is reviewed.
