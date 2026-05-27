# v2 solver root-cause and forward risk audit

Generated: 2026-05-27T07:04:46Z

**root_cause_status:** `PIPELINE_AUDIT_VERDICT_MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED`

## Leading confirmed diagnosis

- True acoustic seed remains valid under unregularized physical v2 replay.
- Prior filtered diagnostic EPS succeeded only after ST A-shift regularization at sigma≈seed.
- Returned Ritz vectors are not operator-consistent with replay; lambda≈1 artifacts dominate.
- Baseline branch recovery has not succeeded; failure is EPS ST/mapping/selection, not loss of v2 coupling.

## ST retry control flow (from code)

**Key question answer:** Yes for prior filtered/unfiltered diagnostic: solver sat on sigma≈seed, LU failed or struggled unregularized, then accepted st_a_shift_frac=0.001 before exhausting all unregularized offset sigmas. New config forbids that path.

### Default production retry order

- outer: stiff_frac in (1e-3, 0.0, 5e-3, 2e-2)  # A-shift tried BEFORE zero
- middle: reg_frac in mass-reg ladder (0, 0.03, ...)
- inner: try_hz in sigma_hz_list

### Unregularized-offset diagnostic order

- stiff_frac in (0.0,) only
- reg_frac in (0.0,) only
- try_hz in offset sigma ladder only

## Invalid audit method (rejected)

- **Method:** Setting st_a_shift_frac while solve_evp=False
- **Reason:** ST regularization is applied only inside _slepc_try_eps_st_setup on ST copies. Replay assembly never constructs the regularized ST operator; such a script cannot test whether ST regularization caused EPS failure.

## Evidence scope

### Confirmed from local code

- Native STSINVERT: lam_phys=mu, eps_eigenvalue_semantics=slepc_backtransformed.
- legacy_double_shift_mapping_disabled=True for native sinvert harvest.
- eps_diagnostic_preserve_all_nconv_candidates saves every converged vector before filters.
- ST A-shift modifies only ST factorization copy; replay uses unregularized GNHEP.
- PGNHEP/purification ruled out when has_EPS_ProblemType_PGNHEP=False on VM.

### Reported from VM operator evidence

- Prior regularized filtered diagnostic superseded for branch verdict.
- True acoustic seed valid at ~243.075 Hz under unregularized physical v2 replay.
- Pre-mapping-fix unregularized-offset solve (7 harvested modes, all xH_Mx=0) is not a valid test of corrected lam_phys=mu mapping; prior seven-mode harvest is not failure evidence.
- Mapping-corrected baseline on VM: nconv_marked=56, preserve_all kept=56, but num_vectors_saved=0 (worker rt/rb filter dropped rows; bank not bridged to config).
- Verdict: MAPPING_FIXED_UNREGULARIZED_BASELINE_CANDIDATE_PERSISTENCE_FAILURE — inconclusive, not ST failure, not Stage-2 trigger.
- PGNHEP/purification: not_justified_use_nullspace_reduction_plan in current VM environment.
- PASS replay recertification: some_modes_valid_physics_wrong_frequency_labels_only.
- Pre-mapping unregularized-offset verdict (historical): UNREGULARIZED_OFFSET_OUTPUT_OR_REPLAY_INCONSISTENT.
- Saved-vector mass-norm audit classification: EPS_RETURNED_ONLY_MASS_NULL_CANDIDATES_IN_UNREGULARIZED_SOLVE.
- Mapping-corrected baseline verdict: MAPPING_FIXED_UNREGULARIZED_BASELINE_OUTPUT_OR_REPLAY_INCONSISTENT.
- Replacement persistence-fixed baseline EPS already ran on VM; 56/56 candidates persisted.
- Replacement evaluator headline: MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED.
- Full pipeline audit verdict: MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED (all_persisted_sparse_mass_null_pre_sparsify_dense_not_preserved).

### Requires VM runtime artifact evaluation


## Production exposure

- **replay_only_true_seed_audits:** proven protected (no EPS/ST)
- **paths_using_eps_with_default_st_ladder:** potentially exposed; report-only spot-check possible
- **prior_PASS_L0_material_mesh_convergence:** not invalidated automatically; smallest spot-check after baseline closed
- **standard_seeded_retrieval_with_st_a_shift_frac_0.001:** potentially exposed (VM operator evidence)

## VM filtered evaluation (runtime, optional)

**Loaded:** verdict `FILTERED_DIAGNOSTIC_NO_PHYSICAL_BRANCH_RECOVERED`

## Finite closure plan

- **next_allowed_action_after_VM_report:** Persisted sparse vectors mass-null; dense pre-sparsify not preserved. Review architecture audit and lossless persistence preflight; no ST failure verdict. No additional EPS until lossless path approved.
- **maximum_additional_baseline_solves_before_escalation:** 1
- **maximum_additional_code_fix_cycles:** 0

**Blocked actions:**
- hole_radius_large
- mesh_convergence_resume
- L_prod
- L_check
- LHS
- v2_production_promotion
- PGNHEP_purification_in_current_VM
- another_sigma_adjustment
- another_filter_only_EPS_rerun
- another_ST_mapping_variant
- stage_2_nullspace_reduction

**single_permitted_next_action:** `None`
**unregularized_offset_solve_completed:** `None`
**unregularized_offset_evaluation_verdict:** `None`
**saved_vector_mass_norm_classification:** `EPS_RETURNED_ONLY_MASS_NULL_CANDIDATES_IN_UNREGULARIZED_SOLVE`

## Forward Risk Register

| risk_or_mismatch | where_in_code | evidence_source | already_triggered_or_only_possible | impact_if_unfixed | how_to_detect_before_next_solve | minimal_fix_required | blocks_hole_radius_large? | blocks_mesh_convergence_resume? | blocks_v2_production_promotion? |
|---|---|---|---|---|---|---|---|---|---|
| lambda≈1 sigma/mapping artifacts from regularized ST at sigma≈seed | fem_main_3d._slepc_try_eps_st_setup + _slepc_physical_lambda | local_code\|VM_operator_evidence | already_triggered (filtered diagnostic) | False branch recovery; reported f≈sigma but replay lambda≈1 | Replay Rayleigh lambda; reject abs(lambda-1)<=tol; require op-consistent ST | Unregularized-offset sigma ladder; forbid ST reg for diagnostic verdict | True | True | True |
| EPS ST-regularized solve not replay-consistent | ST copy A_st vs harvest/replay GNHEP A | confirmed_from_local_code | already_triggered | Candidates are eigenvectors of perturbed ST, not physical GNHEP | Require diagnostic_operator_consistent_with_replay and st_*_frac==0 | diagnostic_requires_unregularized_ST fail-closed policy | True | True | True |
| reported frequency vs replay inconsistency | Harvest f_hz vs replay Rayleigh f_hz | VM_operator_evidence | already_triggered | Accept spurious sigma-near modes | reported_vs_replay_frequency_consistent gate | Post-harvest physical filter (already in v2_seed_branch_candidate_filter) | True | True | False |
| p_frac-only branch selection | mode_diagnostics p_frac_energy_phys | confirmed_from_local_code | already_triggered | High p_frac on non-physical vectors | MAC+replay+freq gates for branch verdict | Already enforced in diagnostic evaluate | True | True | False |
| production paths accepting ST-regularized modes without replay screen | fem_main_3d default stiff_ladder | confirmed_from_local_code | only_possible | Prior PASS may include sigma artifacts | Report-only replay filter on saved modes | Spot-check after baseline diagnostic closed | False | True | True |
| stale reports overriding newer evidence | v2_mesh_convergence/diagnostics/*.json | confirmed_from_local_code | only_possible | Premature resume/promotion | mesh_convergence_may_resume=False until unreg-offset verdict | Supersede prior diagnostic verdicts explicitly | True | True | True |

**mesh_convergence_may_resume:** `False`
**mesh_convergence_pass:** `Pending`
**v2_production_promotion_ready:** `False`
**lhs_promotion_blocked:** `True`

