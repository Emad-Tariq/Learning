# Gallery-aware accidental-match calibration test

This experiment tested whether a candidate identity's best cosine match should be judged relative to the accidental-match distribution for its gallery size and viewpoint coverage. The null distributions used different-identity photos from the fit role only. The empirical percentile replaced raw cosine as the candidate score. Both methods placed `new_individual` first or fifth using a method-specific threshold selected on development seed 2031. Seeds 2036–2038 were then checked without changing those thresholds.

| Episode | Raw cosine MAP@5 | Gallery-calibrated MAP@5 |
| --- | ---: | ---: |
| Development 2031 | 0.8436 | 0.7702 |
| Check 2036 | 0.8524 | 0.8104 |
| Check 2037 | 0.8565 | 0.8009 |
| Check 2038 | 0.8557 | 0.7974 |

On the three check episodes, calibrated known-query MAP@5 was 0.6921, 0.7007 and 0.6962; the raw baseline scored 0.7470, 0.7569 and 0.7732. Calibrated unknown-query scores also fell in every episode. The direct empirical-percentile approach is therefore rejected.

The open-set episodes contain known queries from identities with gallery photos and unknown queries from held-out identities of the same species. Unknown identities contribute at most two query images each. The simulated unknown fraction is near one half; the true competition fraction is unavailable, so these MAP@5 values are not leaderboard estimates. The comparison is still informative because both methods used the same episodes and threshold-selection procedure.

This experiment rules out the tested percentile formulation, not every possible gallery-aware likelihood model. In particular, percentiles can compress many strong candidates near 1 and discard useful differences between their raw scores. Any later variant needs a new predeclared test set; seeds 2036–2038 have now been inspected.

Run `python gallery_calibration_test.py` from this directory with the downloaded data in `~/Downloads/reid_student_data_main`. The script uses the shared Model 6 split utilities and the supplied global descriptors. It does not create a competition submission.
