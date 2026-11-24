# Abhay — Model 5: Open-Set Score Fusion (OSSF)

## Implementation

- Adapted Dalvin's Model 3b retrieval and LightGBM components into an importable baseline; source attribution and hash are retained in `provenance.json`.
- Implemented final-stage logistic fusion of unknown-detector, candidate-ranking, similarity, viewpoint and SIFT evidence.
- Added legacy-rule and detector-only ablations on identical candidate identities and conditional identity scores.
- Separated identities across sequential stacking stages, calibration and outer validation.
- Added explicit feature-ID mapping, role-local SIFT caching, split manifests, checkpointing and per-query diagnostics.
- Added Kaggle and Windows notebooks with embedded executable source, plus synthetic integration tests.

## Measured result

The supplied completed-run output contains 506 validation queries. Dalvin's rule achieves 0.916206 MAP@5, detector-only 0.909947 and Model 5 0.911594. Fusion delta is -0.0046113306982873414. Known-query performance improves slightly overall; unknown-query performance decreases. No overall improvement or Kaggle leaderboard result is claimed. `RESULTS.md` records the species breakdown, limitations and interpretation.

## Verification

Three synthetic tests passed: compressed SIFT discovery; complete three-round training/evaluation with extracted sibling SIFT folders and submission generation; and shuffled feature-ID alignment. Notebook cells compile and embedded Python modules match the standalone source. The Windows notebook's preflight was executed against the downloaded dataset: 7,163 training sightings, 2,791 test sightings and a 9,954 x 2,152 global feature matrix, with both SIFT sources present.

The original local environment produced a SciPy/NumPy compatibility warning. `requirements.txt` constrains NumPy below 2.5 for a compatible installation. Synthetic verification establishes execution and invariants, not model accuracy.

## Experiment provenance

Recorded date: 24 November 2026. Results were transcribed from the supplied notebook output. Full run metadata and the Kaggle version identifier were not included; these remain unrecorded rather than inferred. Runtime artifacts and datasets are excluded from Git. The copied baseline remains credited to Dalvin.
