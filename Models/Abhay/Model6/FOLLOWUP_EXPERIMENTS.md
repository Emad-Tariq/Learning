# Matching follow-up, 24 November 2026

All experiments used the same identity-disjoint seed-2026 split and whole near-duplicate gallery/query groups as Model 6. Original-cosine MAP@5 was 0.8457 on development (215 queries) and 0.9091 on evaluation (271 queries). These are closed-set matching tests, not competition scores. No variant below is recommended for submission.

| Idea | Development observation | Fixed-choice evaluation observation |
| --- | --- | --- |
| Pairwise MLP (Model 6) | 0.2625 MAP@5 | 0.3046 MAP@5 |
| Gallery hub penalty, mean of 30 wrong-identity nearest neighbours, weight 0.25 | Slight gains in both species | 0.9084 vs 0.9091 cosine |
| View-conditioned impostor z-score, statistics fitted on fit identities | Gains in both species | 0.9041 vs 0.9091 cosine |
| 384-dimensional PCA with partial whitening, fitted on fit images | 0.8636 vs 0.8457 cosine | 0.8975 vs 0.9091 cosine |
| RootSIFT bag of 64 visual words with TF-IDF, fused at weight 0.3 | Gains in both species | 0.9082 vs 0.9091 cosine |
| Linear triplet ranking on symmetric pair features | At best about 0.18 MAP@5 | Not evaluated after development failure |
| Mean pooled RootSIFT descriptor | Near tie with cosine | Not evaluated after development tie |

The repeated reversals show that this evaluation split can distinguish promising development gains from gains that do not transfer. Since several choices have now been inspected against its outcomes, it must no longer be treated as an untouched final test for future model selection. A subsequent model should use new, predeclared identity partitions and report performance over several seeds, with one reserved final partition or competition validation held back until the method is frozen.

## Error pattern

Development cosine top-one failed on 39 of 215 queries. Leopard frontal views were especially hard: 5 of 15 front queries were correct at rank one, compared with 46 of 52 left views. Several wrong identities recur among the top matches. This points to viewpoint robustness as a worthwhile direction, but no tested correction has yet produced a credible held-out improvement.

The exploratory commands were run against the local supplied MIEW-ID and SIFT files. Only Model 6's full pipeline and outputs were saved; the follow-up variants were prototypes. Their numbers should be treated as screening results until reproduced in a versioned script on fresh splits.
