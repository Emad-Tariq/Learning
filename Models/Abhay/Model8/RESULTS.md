# Model 8: union shortlist — completed full-pipeline comparison

The union did not pass the predeclared overall gate. Overall MAP@5: 0.934272 → 0.934910; difference +0.000638. The descriptive 95% identity-cluster bootstrap interval for the paired difference is [-0.002416, +0.003491]. Keep Dalvin's baseline as the default: this run does not establish a reliable improvement. The interval reflects these fixed models and validation identities; it does not measure retraining variance. The earlier EDA examined the labelled dataset, and episodes reuse queries, so this is not an untouched final evaluation or a leaderboard claim.

The union newly includes the correct identity for 2 query appearances; 0 of those reach the final top five. Candidate recall is unchanged in both the missing-side and small-gallery slices. Their score changes therefore arise from changed model ranking/unknown decisions rather than recovery of an omitted correct identity.

Both variants were retrained through all three stages using 12 simulations per stage. The baseline keeps its original LDA/SIFT top 15; the union retains those and appends distinct raw top-15 identities. All other model settings and role-local SIFT evidence are shared. No nuisance transformation is used.

| Slice | Appearances | Unique queries | Baseline MAP@5 | Union MAP@5 | Difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 1829 | 1147 | 0.934272 | 0.934910 | 0.000638 |
| Known animals | 991 | 783 | 0.897191 | 0.898873 | 0.001682 |
| Unknown animals | 838 | 663 | 0.978123 | 0.977526 | -0.000597 |
| Missing side (known) | 49 | 37 | 0.560204 | 0.571088 | 0.010884 |
| Same side available | 659 | 518 | 0.950405 | 0.950051 | -0.000354 |
| 1–3 gallery photos (known) | 134 | 100 | 0.691169 | 0.695522 | 0.004353 |
| 4+ gallery photos (known) | 857 | 683 | 0.929405 | 0.930669 | 0.001264 |
| Missing side + 1–3 photos | 41 | 30 | 0.542683 | 0.567886 | 0.025203 |
| Viewpoint missing | 64 | 37 | 0.822656 | 0.820052 | -0.002604 |

## Candidate coverage

Average valid union candidates: 19.61; range 16–25. Newly included correct candidates: 2 query appearances (2 distinct photos). Of those, 0 finish in the final top five. See `slice_results.csv` for candidate recall by slice.

## Validation episodes

| Episode | Baseline MAP@5 | Union MAP@5 | Difference |
| --- | ---: | ---: | ---: |
| 22052 | 0.929718 | 0.930595 | +0.000877 |
| 22053 | 0.938067 | 0.939468 | +0.001401 |
| 22054 | 0.935255 | 0.934898 | -0.000357 |

## Baseline parity and limitations

```json
{
  "episode": 22052,
  "queries": 627,
  "identical_top5": true,
  "max_abs_final_pnew_difference": 0.0,
  "max_abs_candidate_score_difference": 0.0,
  "reference_checkpoint": "C:\\Users\\SANGEETA-DAILY\\Documents\\ChatGPT\\Animal-ReID\\Models\\Abhay\\Model7\\local_runs\\paired_2052\\paired_models.joblib",
  "reproduces_saved_predictions": true
}
```

Synthetic tests verify feature parity with the preserved Dalvin baseline, exact union membership and removal of padded rows from fitting/scoring/ranking. Baseline checkpoint parity is audited separately on validation episode 22052. Bootstrap intervals cluster appearances by identity and stratify by species; targeted slices can contain few independent animals. The protocol uses identity-disjoint training stages and smaller galleries than final competition inference. The result compares complete pipeline components under this protocol, not the original full-data competition training. No submission or baseline replacement is performed.

Detailed per-query predictions, episode splits, configuration, source hashes and models are retained in `local_runs/union_2052`. Reproduce with `union_eval.py`, then `audit_baseline.py` and `summarize.py`.