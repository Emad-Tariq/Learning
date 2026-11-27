# Model 7 paired open-set validation

The nuisance transformation gives a small pooled gain, but does not pass the predeclared submission gate. It improved two of three validation episodes and slightly worsened the third. No submission CSV was generated or submitted to Kaggle.

These are local validation scores from an identity-disjoint comparison of Dalvin Model 3d components. They are not Kaggle leaderboard scores. The team's reported best Kaggle score remains 0.85078.

| Method | Overall MAP@5 | Known-animal MAP@5 | New-animal MAP@5 | Top-one accuracy |
| --- | ---: | ---: | ---: | ---: |
| Dalvin components | 0.934272 | 0.897191 | 0.978123 | 0.901039 |
| Nuisance raw evidence | 0.935721 | 0.899865 | 0.978123 | 0.902679 |

| Validation episode | Baseline MAP@5 | Nuisance MAP@5 | Difference |
| --- | ---: | ---: | ---: |
| 22052 | 0.929718 | 0.933121 | +0.003402 |
| 22053 | 0.938067 | 0.939300 | +0.001232 |
| 22054 | 0.935255 | 0.934898 | -0.000357 |

The pooled gain is +0.001449 MAP@5, or about 0.145 percentage points. Known-animal matching improves by 0.002674; pooled new-animal MAP@5 is unchanged. The species breakdown is slightly negative for hyenas (−0.000381) and positive for leopards (+0.002275). This does not establish a reliable improvement over the team's full competition model.

## Reliability

There are 1,829 query appearances across the three episodes, representing 1,147 unique images and 105 queried animal identities. The episodes share the same outer validation identities and can reuse images, so they are correlated. A bootstrap that resamples complete identities gives an approximate 95% interval for the paired improvement of −0.001674 to +0.004569. This interval includes zero. It is conditional on this fitted model pair and outer split, not a confidence interval for a Kaggle gain.

Both versions used the same identity roles, simulated episodes, candidate identities, SIFT matches, LightGBM settings and seeds. The nuisance directions used only the separate metric-fitting identities. The last models were saved before validation scoring. The independent output audit passed: all training identities have one role; unknown query identities are absent from their episode gallery; known query identities are present; predictions contain valid same-species gallery labels; and all per-query and pooled MAP@5 values recompute correctly from the prediction strings. Three integration tests also passed for baseline feature parity, fitting-label separation and resumable SIFT cache saves.

## What the test supports

The earlier raw-cosine matching gain does not translate into a substantial gain once Dalvin's stronger retrieval, SIFT verification and learned open-set ranking are included. The proposed feature transformation may add a small amount of information, mainly for leopard matching, but the present evidence is too weak to promote it as the new best model. The interval includes a small loss, one episode worsens, and the predeclared gate therefore fails. Its thresholds were not changed after scoring.

This outer-split protocol uses smaller training galleries than the teammate's original full-data simulation protocol. It tests the complete components under controlled identity separation, not the exact trained model behind the reported public score. Source identities also appeared in earlier research; a new partition prevents supervised fitting overlap but does not create an entirely new independent dataset.

## Reproduce

From this directory with the supplied metadata in Downloads and SIFT directories alongside it:

```powershell
python paired_eval.py --workers 8 --predict-test
python audit_results.py --run local_runs/paired_2052
python -m unittest test_integration.py
```

The measured run used seed 2052, three stacking stages, 12 simulations per stage, 150 SIFT neighbours, 15 candidates and 400 candidate-scorer trees. The detector retains the original 300 trees. The transform uses 16 directions at strength 0.5, selected before this experiment. Model fitting and validation took about 45.2 minutes on this PC, primarily mutual SIFT verification. Completed pairs are now cached and reruns reuse them.

`results/paired_metrics.json`, `results/paired_audit.json` and `results/paired_validation.csv` preserve the measured comparison. Runtime models, episode assignments and mutual SIFT cache are under `local_runs/paired_2052`. Both notebooks execute the same paired test and output audit. Nothing was staged, committed or pushed.
