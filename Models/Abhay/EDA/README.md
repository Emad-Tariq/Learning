# Animal ReID exploratory data analysis

Analysis date: 25 November 2026. Abhay built reproducible metadata, embedding, retrieval and geometric-match diagnostics with CSV tables, plots and an HTML report. Start with [design findings](DESIGN_FINDINGS.md) for the observations used in the model documentation.

## What the EDA contributes

The main design insight is to distinguish missing reference-side coverage from genuine new-animal evidence. Sparse albums, long-tailed identity frequencies, near duplicates and uneven viewpoint coverage affect retrieval and the interpretation of weak matches. Photo-weighted and identity-weighted scores can differ substantially. The diagnostic findings are descriptive, with small and overlapping slices; they do not establish a causal explanation or a proven correction.

The later Models 8–10 tested candidate expansion, coverage corrections and difficult training episodes. Their results did not establish a reliable fix. Models 5–7 preceded this EDA report, so these later observations are retrospective context for those earlier experiments.

## Evidence and reproduction

`identity_counts.csv`, `identity_macro_scores.csv`, `retrieval_summary.csv`, `retrieval_slices.csv`, `lda_side_coverage_summary.csv`, `candidate_union_recall.csv`, `sift_sample_summary.csv` and `duplicate_groups.csv` support the design findings. Scripts and original derived artifacts remain in this directory. `artifact_manifest.json` records hashes of the original generated artifacts; the newly written documentation is outside that historical manifest.

Install the recorded requirements and run from the repository root with the original supplied data in the script-configured locations:

```powershell
python -m pip install -r Models/Abhay/EDA/requirements.txt
python Models/Abhay/EDA/run_eda.py
python Models/Abhay/EDA/check_lda.py
python Models/Abhay/EDA/finish_report.py
```

The scripts write derived artifacts beside themselves. Near-duplicate groups approximate encounters using within-identity cosine similarity; original encounter IDs are unavailable. Eligible known-query diagnostics require a separate usable reference, so they exclude some sparse identities. PCA/LDA diagnostics do not reproduce the full SIFT/LightGBM pipeline. Treat simulated thresholds as exploratory diagnostics, not independently calibrated deployment thresholds.
