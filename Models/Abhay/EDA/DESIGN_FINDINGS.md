# EDA findings that inform model design

Abhay implemented reproducible metadata, embedding, retrieval and geometric-match analysis, with CSV summaries, plots and an HTML report. The observations below concern the labelled data and diagnostic gallery/query episodes. They are descriptive findings rather than causal explanations or final model performance claims.

| Observation | Design implication | Experiments |
| --- | --- | --- |
| Of 345 leopard identities, 204 have at most three photos and 99 are singletons. | Diagnose small reference albums and avoid letting well-photographed identities dominate evaluation. | Models 9–10, 12, 15 |
| Raw leopard retrieval scores 0.8862 when photo-weighted but 0.5996 when eligible identities receive equal weight. | Report identity-weighted views and performance slices alongside pooled scores. | Coverage diagnostics and later evaluations |
| In one group-separated PCA/LDA diagnostic, same-side-present versus absent MAP@5 was 0.9310/0.4750 for hyenas and 0.9561/0.1969 for leopards. | Missing-side coverage can explain weak known-animal evidence; distinguish it from genuine novelty. | Models 8–10 and 15 |
| Raw and projected retrieval can propose different candidates. | Measure candidate omission separately from errors in ordering or novelty insertion. | Model 8 union shortlist |
| In selected pair samples, strong SIFT support was much less common across opposite flanks. | Combine geometric support with coverage evidence; a weak geometric match alone does not prove novelty. | Models 9–11 and 15–17 |
| Near duplicates and repeated photos can make retrieval appear easier. | Keep approximate encounter groups intact and split supervised roles by identity. | Pair verification, simulated episodes and audit tools |

The side-absence slices contain only 22 hyena and 48 leopard queries in that diagnostic. SIFT comparisons used selected samples, not random population estimates. Near-duplicate groups approximate encounters because original encounter IDs are unavailable. Singletons cannot supply a known query and a separate reference, so eligible-query diagnostics do not represent every animal.

These findings motivated hypotheses. Models 8–11 did not demonstrate that larger shortlists, residual coverage corrections, difficult episodes or MLP calibration reliably solved the problem. Model 12's gain concerned a different intervention: larger-gallery training. Later album, graph and joint-scoring experiments preserve those distinctions.

Evidence: `identity_counts.csv`, `identity_macro_scores.csv`, `retrieval_summary.csv`, `retrieval_slices.csv`, `lda_side_coverage_summary.csv`, `sift_sample_summary.csv`, `duplicate_groups.csv`, and the analysis scripts in this directory. The full EDA dates to 25 November 2026; it must not be attributed as the original motivation for Models 5–7.
