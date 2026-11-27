# Model 11: full-vector pair MLP for new-animal detection

Open `MLP_NewAnimal_Kaggle.ipynb` in Kaggle, attach the complete competition
input (metadata, MIEW-ID features/IDs, both SIFT sources), select CPU and Run All.
No additional code dataset is needed: source files and tests are embedded.
Kaggle's existing libraries usually suffice; the optional install switch is off.
No pretrained weights, images, PyTorch or attention layers are used.

This is a hypothesis test. The completed local run did **not** improve overall
MAP@5 or preserve actual-new-animal detection; see `RESULTS.md`.

## Fixed protocol

1. Allocate identities separately by species with seed 2052: three disjoint
   20% baseline training stages, 20% calibration and 20% validation.
2. Fit Dalvin's preserved baseline with 12 episodes per stage, 15 candidate
   identities, 150 SIFT neighbours, 400-tree candidate scorers and 300-tree
   unknown detectors. PCA/LDA, RootSIFT and all stacking logic are unchanged.
3. From the baseline training identities, allocate 80% to pair fitting and 20%
   to pair early stopping, by identity. Baseline training uses the full 60% group;
   no supervised classifier fits on calibration/validation identities at this step.
4. Sample same-species pairs: cross-near-duplicate-group positives, hard
   cosine-neighbour negatives and random negatives. Cap anchors per identity,
   deduplicate unordered pairs, and cap at 24,000 fit / 8,000 stopping pairs.
   Near-duplicate groups use raw cosine >=0.97 within each fitting identity.
5. Train a plain scikit-learn MLP with 32/16 hidden units on symmetric full-vector
   absolute differences, elementwise products and cosine (4,305 inputs). Standard
   scaling fits pair-training data only; transformed values are clipped at six.
   Fixed L2=1, learning rate=0.0003, at most 40 epochs; select stopping log-loss
   minimum using only pair stopping identities, with patience six.
6. For each query and each of the 15 baseline candidates, score every reference
   photo. Aggregate max, top-two/top-three means, mean and fraction above 0.5.
   These are evidence features, not calibrated query probabilities. Gallery
   counts, candidate-score margins and baseline context enter the calibrator.
7. Fit two logistic calibrators (C=0.1) on six open-set episodes from calibration
   identities. One uses baseline summaries; the other adds MLP evidence. Labels
   indicate whether the query's true identity is absent from that episode gallery.
8. Freeze everything, then evaluate three validation episodes. Primary prediction
   uses the fixed 50/50 average of baseline P(new) and the MLP calibrator P(new).
   Keep baseline candidates, known-only probabilities and final ranking formula.
   Report summary-only 50/50 blending and full MLP replacement as controls.

Calibration and validation whole-identity unknown holdouts and gallery/query
near-duplicate grouping use the preserved baseline protocol. No query truth or
identity name enters pair features or calibration features. Identity roles,
pair fitting/stopping roles, score histories, config/source hashes, checkpoints
and per-query outputs are saved. Never compare a reduced run to the full run as
evidence of a quality gain. Change OUTPUT when changing code/settings.

## Acceptance and interpretation

The primary gate requires positive pooled and each-episode MAP@5 differences,
no decrease in known MAP@5, unknown MAP@5 or unknown top-one detection, and a
positive identity-cluster bootstrap lower bound. Never tune blend settings on
validation or silently switch to a better-looking diagnostic arm. Failure retains
the baseline. Passing would still need independent confirmation.

The default allocation and labelled data have been examined in prior work; this
is development evidence. Repeated episodes share photos/identities. Species
stratified bootstrapping clusters by true identity and does not include retraining
variance. Role galleries are smaller than competition inference galleries.

## Local commands

```powershell
python -m unittest discover -s Models/Abhay/Model11 -p test_verifier.py -v
python Models/Abhay/Model11/run.py --data PATH_TO_METADATA --output NEW_OUTPUT
python Models/Abhay/Model11/audit_results.py NEW_OUTPUT
```

`--reuse-local` is a local-only speed option for the existing original seed2052,
12-simulation, 400-tree checkpoint and verified source caches. It is not used in
the Kaggle notebook. The saved local run uses this option and independently
verifies all baseline predictions against the source output.

## Generate a competition CSV

The updated notebook creates `/kaggle/working/model11_checkpoint_bundle.zip`
after training and refreshes it after test prediction. It contains the models,
calibrators, SIFT cache, configuration, metrics and exact sources, with per-file
checksums. Download this ZIP or preserve it in saved notebook output before
ending the session: files left only in `/kaggle/working` are temporary.

In a fresh session, attach the ZIP and competition data, set `MODE='submission'`,
and Run All. Training and validation are skipped. If more than one backup is
attached, set `RESTORE_ARCHIVE` explicitly. Training mode defaults to
`GENERATE_SUBMISSION=False` so a training run finishes at the validation/backup
step. Set it true to also generate the CSV in that same run. Submission mode
always runs test prediction. Previously cached SIFT pairs are reused; additional
full-gallery pairs still need computing once. Keep the refreshed ZIP afterward.

For an already-completed Kaggle run, paste the entire contents of
`Kaggle_Submission_Cell.py` into a new cell at the bottom and run it. The updated
notebook also includes this as an optional last cell. Existing checkpoints under
`/kaggle/working/model11_results` and sources under `model11_code` are required.
This does not rerun training. It builds the full train/test SIFT population,
uses all training photos as gallery, and predicts the unlabelled test photos.
That full-gallery SIFT step can be substantially slower than validation.

The outputs are `model11_results/submission_mlp_blend.csv` (the requested fixed
MLP blend) and `submission_baseline.csv` (control). Both are checked for exact
sample ID coverage/order, five distinct labels and species consistency. Download
the MLP CSV from Kaggle's output files and upload it on the competition Submit
Predictions page. The notebook never submits automatically. The frozen models
and calibrators are used without refitting; gallery-size shift remains a limit.

Dalvin retains attribution for `dalvin_core.py` and `data_protocol.py`, copied
unchanged from the preserved Model 7 baseline. Pair sampling builds on concepts
in Abhay Model 6; this experiment uses full vectors, identity-based early stopping,
query-level open-set calibration and a complete shared-baseline comparison.
