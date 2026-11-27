# Initial matching-only test

Run: local supplied training data, seed 2026, default configuration in `matching.py`. Identity-disjoint roles: 330 fit identities, 110 development identities and 110 evaluation identities. The pair verifier trained on 24,030 sampled pairs for 46 epochs. Near-duplicate groups were kept wholly in gallery or query. There were 215 development queries from 92 identities and 271 evaluation queries from 89 identities. Identities with one group were gallery-only.

| Role | Method | MAP@5 | Top-one | Top-five |
| --- | --- | ---: | ---: | ---: |
| Development | Original cosine | 0.8457 | 0.8186 | 0.8930 |
| Development | LDA cosine | 0.8534 | 0.8233 | 0.9070 |
| Development | PIV MLP | 0.2625 | 0.1349 | 0.5023 |
| Evaluation | Original cosine | 0.9091 | 0.8856 | 0.9446 |
| Evaluation | LDA cosine | 0.9056 | 0.8782 | 0.9483 |
| Evaluation | PIV MLP | 0.3046 | 0.1697 | 0.5535 |

Evaluation MAP@5 by species: hyena cosine 0.9537, LDA 0.9452, PIV 0.3823; leopard cosine 0.8796, LDA 0.8793, PIV 0.2531. The PIV drop occurs in both species. It improves on cosine for only 4 of 271 evaluation queries and worsens 209; the rest tie.

## Interpretation

PIV fails the matching-only decision gate. Its pair classification nearly separates the sampled fit pairs, but its probabilities become extreme on unseen identities. On evaluation queries, the median PIV score for the correct identity and for the strongest wrong identity are both near 1.0. Maximum-over-gallery aggregation then lets an occasional falsely high wrong-identity pair outrank the correct identity. This is evidence of poor transfer from fit identities to new identities and a mismatch between pair-level training and identity-level retrieval. It is an inference from score distributions, not proof of one unique failure mechanism.

The fixed cosine baseline is already strong on this closed-set split. These numbers are not Kaggle leaderboard scores: they exclude `new_individual`, use a smaller gallery, and are computed from training identities. No PIV or baseline configuration was changed after inspecting evaluation results. The next model should target generalizable retrieval with an objective and validation unit aligned to identity ranking, instead of treating this pair probability as a submission score.

Reproduce with `python matching.py --data "$env:USERPROFILE\Downloads\reid_student_data_main" --output ./local_runs/piv_run` from this directory, or run either notebook. The run writes query-level predictions and split assignments for audit. This initial test used the same command with output `local_runs/initial_matching_test` in a separate working copy; runtime outputs are ignored by Git.
