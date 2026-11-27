# Nuisance projection experiment

This experiment learns feature directions that vary between photographs of the same animal, then halves the contribution of the 16 strongest such directions. The remaining dimensions are preserved. Each fitting identity contributes equal total covariance weight, preventing identities with hundreds of photos from dominating. Hyena and leopard transformations are fitted separately using only fit-role identities. No query labels fit the transformation.

The approach is inspired by within-class covariance normalization and nuisance attribute projection, established linear feature transformations. Reference: https://www.isca-archive.org/interspeech_2006/hatch06_interspeech.html . This is a regularized projection experiment, not a reproduction of that paper's complete system.

## Selection and matching checks

Four configurations (16 or 64 directions; suppression strength 0.25 or 0.5) were screened on identity partitions 2041 and 2042. The selected configuration, 16 directions with strength 0.5, was frozen before seeds 2043–2045. It improved both species in all three checks. Three additional partitions 2046–2048 compared it with Dalvin's PCA/LDA retrieval component. The projection improved raw cosine in all six checks.

| Pooled matching checks 2043–2048 | MAP@5 | Top-one accuracy |
| --- | ---: | ---: |
| Raw cosine | 0.858604 | 0.825481 |
| Nuisance-projected cosine | 0.873358 | 0.838752 |

These are 1,507 query appearances across six episodes, not 1,507 independent animals. Identities can recur across partitions. The gain does not establish an unbiased confidence interval or a leaderboard improvement.

Dalvin's stronger PCA/LDA retrieval component remained stronger than either raw or nuisance-projected cosine alone on seeds 2046–2048. Applying the projection before PCA/LDA did not give a consistent gain. Thus nuisance projection is not a proven replacement for Dalvin's model.

## Fixed fusion check

On separate predeclared seeds 2049–2051, compare a fixed equal-weight average of image-level cosine and LDA similarity with an equal-weight average of nuisance-projected cosine and the same LDA similarity. No weights were tuned on those episodes. Maximum image score per identity forms the ranking.

| Pooled checks 2049–2051 | MAP@5 | Top-one accuracy |
| --- | ---: | ---: |
| LDA retrieval alone | 0.896505 | 0.868035 |
| Raw cosine + LDA | 0.895503 | 0.866569 |
| Nuisance-projected cosine + LDA | 0.899218 | 0.870968 |

The nuisance variant beat the raw-fusion variant on all three partitions, but one difference was only 0.000065. It beat LDA alone on two of three. This is a promising feature improvement with a modest additional gain against the stronger retrieval baseline, not evidence of a better full competition model. Dalvin's SIFT verification, learned candidate scorers, and `new_individual` detector were not included in this comparison.

## Reproduction and limits

Run `python nuisance_projection_test.py`, `python nuisance_lda_check.py`, and `python nuisance_lda_check.py --fusion`. Data defaults to `~/Downloads/reid_student_data_main`; scripts import the shared Model 6 split utilities. Each command writes its per-query JSON results in the current directory. PCA/LDA sees unlabeled gallery/query vectors and only gallery identity labels, following the teammate retrieval procedure. Nuisance directions see fit identities only.

The episodes are closed-set: every query has a matching gallery identity. Whole near-duplicate groups remain on one side. They do not evaluate unseen animals. Source identities have also appeared in earlier exploratory work, so these new partitions are robustness checks rather than an untouched new dataset. No configuration can guarantee improvement or zero overfitting. A frozen integration into the full open-set pipeline must pass a direct paired comparison before any Kaggle performance claim.
