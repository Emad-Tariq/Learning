# SIFT coverage audit and paired pipeline check

This experiment tests whether Dalvin Model 3d loses useful evidence because a
shortlisted identity has no SIFT-checked gallery photo. It uses the preserved
three-stage baseline, its identity roles, and its three existing evaluation
episodes. It does not modify the baseline or submit predictions.

## Intervention fixed before retraining

For each query, keep the baseline's 15 candidate identities. For every candidate
with **zero checked gallery photos**, verify up to two reference photos. Select
same-flank photos first when the query shows a left/right flank, then highest
original fingerprint cosine similarity, then global row index to break ties.
Selection does not use query identity labels or correctness. All candidates,
including incorrect candidates and queries of unknown animals, follow this rule.

Keep the original RootSIFT, mutual ratio test, partial-affine RANSAC, neighbour
selection, duplicate grouping and final ranking formula. Recompute SIFT features
with the selected additional query/gallery pairs and add candidate-level checked
photo count, checked fraction and missing-check indicator. Retrain all three
LightGBM stages on the same 12 episodes per stage as the baseline, using the same
400-tree candidate scorer / 300-tree detector settings and seeds. The intervention
combines additional checks and availability features; it does not isolate their
individual effects. No extra query/query SIFT edges are added.

Sparse zero-inlier entries count as **checked**. Reading matrix values alone would
incorrectly conflate failed verification with an unattempted pair. Cached extra
pairs are exposed only in the episode that selects them, so earlier episodes
cannot silently change the baseline or later candidate lists.

## Validation

Before scoring the intervention, reproduce all saved baseline top-five
predictions exactly. Runtime assertions also require unchanged query order,
truth, candidate membership and non-SIFT candidate evidence. Tests check that
stored zero matches retain checked provenance and that changing query truth
cannot change the selected extra pairs or inference features.

The predeclared promotion gate requires a positive pooled difference and positive
difference in each of three episodes, without reducing pooled known-query or
unknown-query MAP@5. Bootstrap intervals resample true identities within species
to account for repeated photos/appearances. These are conditional on fixed fitted
models and existing validation identities, not retraining uncertainty.

## Run

From the project root, with the existing data and preserved baseline artifacts:

```powershell
python -m unittest discover -s Models/Abhay/sift_coverage -p test_coverage.py -v
python -u Models/Abhay/sift_coverage/check_coverage.py --audit-only
python -u Models/Abhay/sift_coverage/check_coverage.py
```

Results are saved under `results/`: `audit_summary.json`, `query_audit.csv`,
`candidate_audit.csv`, `paired_predictions.csv` and `paired_metrics.json`.
Per-episode audit files, cached evaluation frames, a separate SIFT cache and a
resumable intervention checkpoint are also retained.

## Interpretation limits

- These validation identities have been examined in previous EDA and experiments.
  This is a development comparison, not an untouched final holdout.
- Episodes reuse photos and identities. Report appearances and unique counts.
- Identity-disjoint stage/evaluation roles have smaller galleries than full-data
  competition inference. This does not establish the frequency of the gap or a
  leaderboard effect for the original full-data model.
- The audit supplements only candidates with no checks, and at most two photos
  per candidate. It does not test checking every unverified photo of partially
  checked candidates, or every gallery photo of unchecked candidates.
- Seven geometric inliers is an existing evidence threshold, not proof of an
  identity match.

The underlying matching, embedding and stacking implementation remains credited
to Dalvin, as preserved in `../Model7/dalvin_core.py` and `data_protocol.py`.
