# Cursor runtime evidence bundle — lossless adjudication preflight

Generated UTC: 2026-05-27T07:19:24.719089+00:00

Purpose:
- Provide compact VM runtime evidence to Cursor without committing lossy vectors or multi-megabyte generated reports.
- Support clean-adjudication-lane preparation, policy merge correction, and policy-equivalence preflight.
- This bundle is evidence only; it does not authorize a new EPS solve.

Authoritative current status:
- MAPPING_FIXED_UNREGULARIZED_BASELINE_PERSISTED_VECTOR_CONTENT_UNRESOLVED
- Persistence self-test passed.
- Replacement baseline EPS ran once; 56/56 sparse candidate files were persisted.
- Existing sparse candidate vectors are insufficient for ST viability verdict.
- Lossless persistence self-test passed without EPS.
- Runtime provenance shows approved policy values except st_type is missing from prior artifacts.

Intentionally excluded:
- candidate_eps_slot_*.smx.npz
- result_243075.json full source
- multi-megabyte full diagnostic JSON reports
- mesh files
- raw logs
- scratch directories

Original small files copied:
- FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/v2_mapping_fixed_candidate_persistence_self_test.json
- FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/v2_lossless_candidate_persistence_self_test.json
- FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/diagnostics/v2_mapping_fixed_replacement_runtime_provenance_inventory.json
- FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/solves/L_mid/baseline_coupled_v2/seed_branch_recovery_diagnostic_mapping_fixed_unregularized_persistence_fixed/launch_record.json
- FEM/experiments/active_domain_validation/physics_integrity/v2_mesh_convergence/solves/L_mid/baseline_coupled_v2/seed_branch_recovery_diagnostic_mapping_fixed_unregularized_persistence_fixed/sample_spec.json

Generated compact extracts:
- eps_candidate_bank_compact.json
- full_pipeline_audit_compact.json
- replacement_baseline_diagnostic_compact.json
- runtime_policy_field_occurrences.json
