# Mapping-fixed persistence-fixed full pipeline audit

Generated: 2026-05-27T07:04:15Z

**audit_verdict:** `MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED`
**verdict_reason:** `all_persisted_sparse_mass_null_pre_sparsify_dense_not_preserved`
**operator_policy_provenance_mismatch:** `True`

## Conservative policy

- persistence bridge is closed
- self-test passed
- replacement baseline EPS ran once
- 56/56 candidates were persisted (when persistence_closed)
- persisted sparse representations are mass-null under replay when all num_mass_null
- dense pre-sparsify EPS candidate content was not preserved for replay in current run
- current saved vectors are insufficient for a physical ST viability verdict
- no Stage-2 authorization from this evidence
- no additional EPS run authorized in this step

## Persistence status

- self_test_pass (VM evidence design): replacement persistence 56/56 closed
- candidates on disk: 56
- nnz summary: {"count": 56, "min_nnz": 10, "median_nnz": 118, "max_nnz": 1588, "num_nnz_le_100": 22, "num_nonfinite_vectors": 0, "systematic_low_nnz": false}

## Root-cause category

`PERSISTED_EPS_VECTOR_CONTENT_CORRUPTED_OR_TRUNCATED`

Persisted .smx.npz vectors pass through dense_to_csr_f32_column with MODE_VECTOR_RELATIVE_EPS=1e-7 relative to max(|vector|); mass-null replay on all candidates does not prove dense in-memory EPS candidates were mass-null.

## Operator policy (from artifacts)

## Map contract

- u_to_W_found: True
- u_to_W_length: 253587
- u_to_W_crc32: 2457905409
- p_to_W_found: True
- p_to_W_length: 24039
- p_to_W_crc32: 2027087254
- n_reduced_W_expected: 277626
- u_p_map_overlap_count: 0
- u_p_map_union_size: 277626
- map_contract_pass: True
- map_contract_failure_reason: None

- continuation_seed_applied: False
- seed_frequency_hz: 243.0754171175576
- actual_sigma_hz: 243.0754171175576
- st_type: None
- eps_eigenvalue_semantics: slepc_backtransformed
- legacy_double_shift_mapping_disabled: True
- diagnostic_operator_consistent_with_replay: False
- actual_st_a_shift_frac: 0.0
- actual_st_mass_reg_frac: 0.0
- preserve_all_enabled: True
- nconv_marked: 56
- candidate_bank_count: 56
- num_vectors_saved: 56
- serializer_function: fem_mode_array_utils.save_mode_csr(dense_to_csr_f32_column(vec))
- serializer_threshold: MODE_VECTOR_RELATIVE_EPS=1e-7 relative to max(|vector|)
- save_errors: []
- p_to_W_source: v2_sensitivity_solve_result_p_to_W
- p_to_W_length: 24039
- p_to_W_crc32: 2027087254

## Evaluation summary
- num_candidates: 56
- physics_eval_skipped: False
- num_rayleigh_ok: 0
- num_mass_null: 56
- num_branch_recovery_pass: 0
- num_exceptions: 0

## Pipeline control flow (code-derived)
- **EPS_getEigenpair** @ `FEM/scripts/fem_main_3d.py::_slepc_shift_invert_batch`
- **preserve_all_capture** @ `fem_main_3d.py::_slepc_shift_invert_batch (preserve_all_nconv)`
- **worker_row_bridge** @ `fem_main_3d.py worker single-shift path`
- **candidate_persistence** @ `v2_sensitivity_solve.py + v2_mapping_fixed_candidate_persistence.py`
- **candidate_bank_json** @ `v2_mapping_fixed_candidate_persistence.write_eps_candidate_bank_json`
- **mode_energy_summary** @ `v2_sensitivity_solve._save_one (diagnose_mixed_mode, energy participation)`
- **evaluator_load** @ `v2_unreg_offset_report_evaluator._evaluate_one_candidate`
- **replay_rayleigh** @ `physical_fsi_seed_residual_audit._rayleigh_metrics`
- **verdict_assignment** @ `v2_mapping_fixed_baseline_evaluator.assign_mapping_fixed_verdict`
