Emad:
- **Model 1:** Implemented MIEW-ID cosine similarity baseline for species-aware identity retrieval. Achieved 0.53083 public leaderboard score.
- **Model 2:** Extended baseline with SIFT feature matching, RANSAC verification, and species-specific open-set thresholds. Achieved 0.765 public leaderboard score.

Dalvin:
- **Model 3:** LDA-projected MIEW-ID retrieval, RootSIFT + RANSAC verification, and LightGBM open-set ranking (new-individual detector + candidate scorers) trained on simulated open-set splits, with 3 rounds of transductive stacking over nearest test neighbours. Simulated out-of-fold MAP@5: 0.916. Public leaderboard score: pending submission.

Dewang:
- **Model 4:** PCA-whitened MiewID retrieval with viewpoint-aware identity scoring (same-side bonus / opposite-side penalty),
  optional RootSIFT + RANSAC bonus, and a per-species logistic-regression new-individual detector whose probability places
  `new_individual` in the top-5 ranking. Shared open-set validation splits saved to Processed_Data/splits/.
  Public Scoreboard: 0.77431
- **Model 4b:** Upgraded Model 4 into a PCA → LDA retrieval pipeline with LightGBM open-set ranking. Replaced the logistic-regression
  detector with a query-level new-individual detector plus closed-set and all-query (query, candidate) scorers, trained on 16
  regime-diverse simulated open-set episodes (identity-disjoint, near-duplicate-aware group holdout, varying unknown fraction,
  query cap and linking threshold). Added 3 rounds of transductive stacking (each round's predictions for a query's nearest test
  neighbours feed the next round), richer features (same-view / same-flank similarity, per-identity viewpoint coverage, per-identity
  SIFT support counts, candidate centroid similarity, query-to-query embedding and SIFT neighbours), and mutual nearest-neighbour
  RootSIFT matching with RANSAC verification (SIFT pair results cached for reuse). Regularised LDA (shrinkage 0.3) and subsampled
  LightGBM used to limit overfitting. Test inference replays the same stacking rounds with models trained on all simulated episodes.
  Simulated out-of-fold MAP@5: pending run. Public leaderboard score: pending submission.
