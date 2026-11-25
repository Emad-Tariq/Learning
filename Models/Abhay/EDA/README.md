# Animal ReID exploratory data analysis

Analysis date: 25 November 2026. This directory contains the completed EDA of the supplied MIEW-ID features, metadata and RootSIFT files. It includes aggregate tables, plots, query-level diagnostics and reproducible scripts. It does not contain the original feature arrays or SIFT descriptors.

Start with [the full report](report.html) or [the findings figure](findings.png). The report embeds its overview plot and opens independently in a browser.

## Main actionable insight

**Treat missing viewpoint/side coverage as uncertainty about a known animal, and measure it separately from genuine new-animal detection.** A weak appearance match or low SIFT support can occur because the gallery shows a different side. Those signals alone do not establish that the animal is new.

In one group-separated diagnostic episode, PCA/LDA retrieval for left/right queries gave:

| Species | Same-side gallery available: MAP@5 | Same side absent: MAP@5 | Queries with side absent |
| --- | ---: | ---: | ---: |
| Hyena | 0.9310 | 0.4750 | 22 |
| Leopard | 0.9561 | 0.1969 | 48 |

These are descriptive associations on small, non-random slices, not causal estimates or full-pipeline scores. Photo count, identity difficulty and viewpoint coverage are confounded. The immediate action is to retain missing-side, exact-view availability and gallery-size slices in every experiment and diagnose candidate omission, identity ranking and new-animal insertion separately. Do not apply a blanket viewpoint boost or lower the unknown threshold based on this table.

Subsequent full-pipeline experiments did not establish that larger shortlists, coverage score corrections or difficult-episode retraining solve this problem. Those results are documented separately in Models 8–10. The EDA identifies an important failure mode; it does not claim a proven fix.

## Other important findings

- Data: 7,163 labelled photos of 550 animals and 2,791 unlabelled test photos; 2,152 features per photo.
- Sparse identities: 204 of 345 leopard identities have at most three photos, including 99 with a single photo. Only identities with another usable gallery photo can enter the known-retrieval diagnostics.
- Image-weighted raw leopard MAP@5 is 0.8862; giving each eligible identity equal weight yields 0.5996. Report both to avoid hiding difficult small identities behind well-photographed animals.
- In the selected SIFT sample, at least seven inliers occurred for 50%/65% of same-side hyena/leopard pairs and 0%/25% of opposite-side pairs. These are deliberately selected samples of 40 pairs per species/category, not population estimates.
- All 9,954 images have SIFT files with consistent descriptor/coordinate shapes. However, 96 have zero keypoints and 155 have fewer than four, preventing the preserved baseline's geometric verification.
- Missing viewpoints: 136 training and 42 test images. Represent missing values explicitly.
- Hyenas account for about 27% of training photos and 44% of test photos. This does not reveal the unknown-animal fraction in test.

## Files to read first

| File | Purpose |
| --- | --- |
| [report.html](report.html) | Full report, figures, tables and limitations |
| [findings.png](findings.png) | Three main diagnostic findings |
| [overview.png](overview.png) | Six-panel overview |
| [composition.csv](composition.csv) | Train/test species counts |
| [identity_counts.csv](identity_counts.csv) | Photos and viewpoints per identity |
| [viewpoint_counts.csv](viewpoint_counts.csv) | Viewpoint distribution; missing values counted separately in the report |
| [identity_macro_scores.csv](identity_macro_scores.csv) | Identity-weighted raw retrieval scores |
| [retrieval_summary.csv](retrieval_summary.csv) | Self-excluded and duplicate-group-excluded raw retrieval |
| [retrieval_slices.csv](retrieval_slices.csv) | Retrieval by view, coverage and gallery size |
| [view_and_gallery_size.csv](view_and_gallery_size.csv) | Same-view coverage within gallery-size strata |
| [lda_episode_summary.csv](lda_episode_summary.csv) | Raw versus PCA/LDA retrieval on one matched episode |
| [lda_side_coverage_summary.csv](lda_side_coverage_summary.csv) | Missing-side comparison for left/right queries |
| [candidate_union_recall.csv](candidate_union_recall.csv) | Candidate inclusion from raw/LDA union; no full-pipeline claim |
| [open_set_episodes.csv](open_set_episodes.csv) | Raw-similarity unknown-detection diagnostics |
| [sift_sample_summary.csv](sift_sample_summary.csv) | Sampled geometric verification by pair category |
| [hardest_queries.csv](hardest_queries.csv) | Difficult queries and strongest wrong matches |
| [summary.json](summary.json) | Machine-readable findings and metadata checks |

Other CSV files preserve per-photo retrieval, duplicate groups, SIFT coverage, sampled pair evidence, test nearest neighbours, episode scores and PCA/LDA queries. `exact_feature_duplicates.csv` contains two training rows from the same identity; no exact feature duplicates span train and test. `artifact_manifest.json` records file hashes.

## Reproduce

Keep the supplied inputs in these default locations, or edit the data-root constants in the scripts:

```text
~/Downloads/reid_student_data_main/
~/Downloads/sift_features_train.npz/
~/Downloads/sift_features_test.npz/
```

The supplied SIFT paths are extracted directories containing NPY files. Install the recorded analysis dependencies, then run from the repository root in this order:

```powershell
python -m pip install -r Models/Abhay/EDA/requirements.txt
python Models/Abhay/EDA/run_eda.py
python Models/Abhay/EDA/check_lda.py
python Models/Abhay/EDA/finish_report.py
```

Scripts read source data and write derived artifacts beside themselves. The full SIFT header scan and sampled verification can take time. The original analysis directory is retained separately in this workspace.

## Interpretation limits

Raw and PCA/LDA retrieval diagnostics exclude the complete SIFT/LightGBM/stacking pipeline. Near-duplicate groups approximate encounters using within-identity cosine >=0.97; original encounter IDs are unavailable. Training photos can recur across diagnostic query/gallery roles and episodes. PCA may see unlabelled episode queries, while LDA uses gallery labels. Simulated open-set thresholds describe score overlap and are not independently calibrated deployment thresholds. Test identities are unavailable. These are exploratory results, not Kaggle scores or untouched final validation.

The EDA originally suggested possible follow-up experiments. Those suggestions in the HTML report are historical hypotheses; consult the later model results before repeating them.
