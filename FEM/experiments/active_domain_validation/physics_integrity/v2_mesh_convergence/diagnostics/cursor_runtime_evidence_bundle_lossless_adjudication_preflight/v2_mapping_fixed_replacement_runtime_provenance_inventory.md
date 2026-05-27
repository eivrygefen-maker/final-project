# Replacement runtime provenance inventory

Generated: 2026-05-27T07:04:18Z

**conflicts:** 0 **missing:** 1

- **continuation_seed_applied**: selected=`True` status=found basis=eps_batch_diagnostics
- **seed_frequency_hz**: selected=`243.0754171175576` status=found basis=result
- **actual_sigma_hz**: selected=`243.5754171175576` status=found basis=eps_batch_diagnostics
- **sigma_used_hz**: selected=`243.5754171175576` status=found basis=eps_batch_diagnostics
- **st_type**: selected=`None` status=missing basis=None
- **eps_eigenvalue_semantics**: selected=`slepc_backtransformed` status=found basis=eps_batch_diagnostics
- **legacy_double_shift_mapping_disabled**: selected=`True` status=found basis=eps_batch_diagnostics
- **diagnostic_operator_consistent_with_replay**: selected=`True` status=found basis=eps_batch_diagnostics
- **actual_st_a_shift_frac**: selected=`0.0` status=found basis=eps_batch_diagnostics
- **actual_st_mass_reg_frac**: selected=`0.0` status=found basis=eps_batch_diagnostics
- **preserve_all_enabled**: selected=`True` status=found basis=eps_batch_diagnostics
- **nconv_marked**: selected=`56` status=found basis=eps_candidate_bank
- **candidate_bank_count**: selected=`56` status=found basis=eps_candidate_bank
- **num_vectors_saved**: selected=`56` status=found basis=eps_candidate_bank
- **serializer_function**: selected=`fem_mode_array_utils.save_mode_csr(dense_to_csr_f32_column(vec))` status=found basis=code_contract
- **serializer_threshold**: selected=`MODE_VECTOR_RELATIVE_EPS=1e-7 relative to max(|vector|)` status=found basis=code_contract
- **p_to_W_source**: selected=`v2_sensitivity_solve_result_p_to_W` status=found basis=bank_pressure_block_mapping
- **p_to_W_length**: selected=`24039` status=found basis=eps_candidate_bank
- **p_to_W_crc32**: selected=`2027087254` status=found basis=eps_candidate_bank
