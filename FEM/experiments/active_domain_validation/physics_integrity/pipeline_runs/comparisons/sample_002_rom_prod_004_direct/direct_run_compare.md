# Direct M4 run comparison

- reference: `/home/vboxuser/final-project/FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_002/runs/sample_002_m4prod2_strict_clean5`
- ROM: `/home/vboxuser/final-project/FEM/experiments/active_domain_validation/physics_integrity/pipeline_runs/guitars/sample_002/runs/sample_002_rom_prod_004`
- match tolerance: **5.0 Hz**

## Mode counts

| metric | value |
| --- | --- |
| reference deduplicated modes | 506 |
| ROM deduplicated modes | 612 |
| count ratio (ROM/reference) | 1.209486166007905 |

## Frequency matching

| metric | value |
| --- | --- |
| matched | 495 |
| unmatched reference | 11 |
| unmatched ROM | 117 |
| reference recall | 0.9782608695652174 |
| abs error median / p95 / max (Hz) | 0.25792699999999513 / 1.2121969000000024 / 4.400866999999948 |
| rel error median / p95 / max | 0.0009073382100994182 / 0.005016783521073264 / 0.018122450592024424 |

## Frequency bands

| band | ref | ROM | matched | recall | med rel err | p95 rel err |
| --- | --- | --- | --- | --- | --- | --- |
| 60-150 Hz | 79 | 106 | 78 | 0.9873417721518988 | 0.0015017197298943794 | 0.007923612612852448 |
| 150-250 Hz | 83 | 95 | 80 | 0.963855421686747 | 0.0010508541437755823 | 0.004971458229223647 |
| 250-350 Hz | 115 | 131 | 112 | 0.9739130434782609 | 0.0011515310732087992 | 0.004002782116386289 |
| 350-450 Hz | 94 | 117 | 91 | 0.9680851063829787 | 0.0007339120056007775 | 0.00316481153876519 |
| 450-550 Hz | 135 | 163 | 134 | 0.9925925925925926 | 0.0005427736355981486 | 0.0034686037153285358 |

## Performance

| metric | reference | ROM |
| --- | --- | --- |
| worker count | 2 | 3 |
| worker phase (s) | 18660.37 | 1542.66 |
| total pipeline (s) | 18660.37 | 1542.66 |
| peak RSS / worker (bytes) | None | 3651870720 |
| worker-phase speedup | 12.096229888633916 | |
| ROM samples per reference sample | 12.096229888633916 | |

## Practical conclusion

| field | value |
| --- | --- |
| information_retention_per_sample | 0.9773732560988158 |
| throughput_gain | 12.096229888633916 |
| estimated_ROM_samples_per_reference_sample | 12.096229888633916 |
| recommendation | **FAVOR_ROM_FOR_DIVERSITY** |

_Under the same compute-time budget, is the loss per lightweight sample outweighed by the increased number and diversity of training samples?_

Reason: throughput_gain_outweighs_moderate_per_sample_loss
