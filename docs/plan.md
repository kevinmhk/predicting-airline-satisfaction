# Phased execution plan

Strategy and evidence: [approach.md](approach.md). Official snapshots: [overview.md](overview.md), [data.md](data.md), and [rules.md](rules.md).

`[x]` means verified complete. `[ ]` means not completed. Optional phases may be skipped with a short reason; do not mark an unrun experiment complete. This is a task list, not authorization to execute it. Measured results are recorded in [approach.md](approach.md); Phase 8 remains pending.

## Phase 0 — Establish facts and boundaries

- [x] Create the local project, Git repository, and public-documentation snapshots.
- [x] Download the official competition files and verify their move into `data/` with before/after SHA-256 checks.
- [x] Restore `/data/` to `.gitignore` and remove the uploaded data commit from branch history; verify a fresh remote clone contains no data history. GitHub caches and existing clones are outside that guarantee.
- [x] Read all saved documentation and inspect train, test, and sample-submission files locally.
- [x] Verify schema compatibility, unique/disjoint train/test IDs, and sample-submission ID order.
- [x] Measure target balance, missingness, feature cardinalities, basic distribution differences, and exact duplicate/overlap hashes.
- [x] Record observations, model proposals, limitations, and compliance questions in `approach.md`.
- [ ] Confirm the user's Kaggle account, competition participation status, and unresolved eligibility/employer/sanctions checks before entry or submission. Do not infer acceptance from a successful download.
- [x] Record the user's confirmation that external AI assistance is permitted for this implementation/training increment. Hosted coding/review assistance was used; this permission is user-confirmed, not independently host-verified. Raw CSVs and model training remained local.
- [x] Obtain authorization for the first implementation increment. Registration, terms acceptance, spending, publication, commit, push, and submission remain unauthorized.

Exit check: observed findings are distinguished from hypotheses, and the next execution scope and access boundaries are explicit.

## Phase 1 — Prepare the smallest reproducible environment

- [x] Inspect current repository instructions and working-tree state before editing.
- [x] Create a project-local virtual environment; do not modify the unrelated RD-Agent environment used for inspection.
- [x] Install and pin pandas, NumPy, and scikit-learn. Confirm the installed scikit-learn API supports the selected categorical boosting path.
- [x] Set bounded CPU thread counts, initially no more than two; check available memory and disk before training.
- [x] Extend `.gitignore` for the virtual environment, Python caches, and a single local `artifacts/` directory while preserving `/data/`.
- [x] Define one training entry point and one lightweight test module. Avoid package scaffolding, configuration frameworks, and orchestration services.
- [x] Store models, metrics, validation predictions, and submission files under ignored `artifacts/`, not `docs/` or tracked source folders.

Exit check: the isolated environment imports required packages, dependency versions are recorded, and Git excludes data/runtime artifacts.

## Phase 2 — Define and test the validation contract

- [x] Implement input checks for required fields, train/test predictor agreement, valid binary target, and unique/disjoint IDs. Return an actionable error for invalid input.
- [x] Exclude `id` and `satisfaction` from predictor inputs. Separate nominal categorical features from numeric features explicitly.
- [x] Create the stratified 80/20 outer split with seed 42. Record split indices locally so models use identical validation rows.
- [x] Check split disjointness, complete training-row coverage, and presence of both classes in each partition.
- [x] Record SHA-256 input fingerprints and exact dependency versions with each run. Leave source data unchanged.
- [x] Define common evaluation output: model/settings, split seed, ROC AUC, log loss, elapsed fit time, and convergence/early-stopping information.
- [x] Write a small synthetic regression check before implementing nontrivial preprocessing/export behavior. Cover missing delay, an unseen category, exclusion of ID/target features, and clean-interpreter joblib reload without using raw competition rows in test fixtures.
- [x] Check that preprocessing is trained only on the fit partition. Reserve the outer validation set for evaluation, not early stopping or transformation fitting.

Exit check: the synthetic check passes, split membership is reproducible, and both candidate models share the same evaluation contract.

## Phase 3 — Measure the baseline

- [x] Compute a constant-probability reference from fit-partition target prevalence; evaluate its ROC AUC and log loss.
- [x] Build the logistic regression pipeline with training-fitted median imputation, numeric scaling, and one-hot categorical encoding with unknown-category handling.
- [x] Run a small smoke fit to verify valid two-class probabilities and the end-to-end metric path before a full baseline fit.
- [x] Train on the full fit partition. Check convergence warnings; no convergence adjustment was required.
- [x] Save outer-validation probabilities with row IDs, metrics, settings, seed, input fingerprints, and elapsed time locally.
- [x] Inspect AUC/log-loss results against the constant reference. The measured baseline is plausible and emitted no convergence or probability errors.
- [x] Record measured results and limitations in `approach.md` without replacing the distinction between observations and proposals.

Exit check: a reproducible baseline has real validation scores and no unresolved fitting or probability-validity failures.

## Phase 4 — Compare histogram gradient boosting

- [x] Build a scikit-learn `ColumnTransformer` with safe `OrdinalEncoder` unknown-category handling and explicit categorical-column markers for histogram boosting. Preserve NaN numeric delays.
- [x] Extend the synthetic check to cover the boosting transformation and stable feature order. Verify missing/unseen values do not cause an invalid prediction.
- [x] Set a small, bounded initial configuration. Use only an internal subset of the fit partition for early stopping.
- [x] Run a smoke fit, then train the first full histogram-boosting contender on the same outer split as the baseline.
- [x] Save aligned validation probabilities, settings, actual iterations, metrics, and runtime under `artifacts/`.
- [x] Compare ROC AUC, log loss, runtime, and observed resource usage against logistic regression. Check major error slices by travel type, customer type, and cabin class; report associations, not causal conclusions.
- [x] Select HGB as the initial nonlinear baseline because its gain repeated on both fixed splits; the later, separately authorized challenger increment is recorded in Phase 5.

Exit check: both models have directly comparable measured results, with a provisional choice and explicit uncertainty.

## Phase 5 — Confirm the choice and run only justified experiments

- [x] Re-evaluate shortlisted candidates on another fixed stratified split seed (2024). Keep the comparison protocol identical across candidates.
- [x] Check whether model ranking and claimed improvements are repeatable. HGB won both fixed splits by more than the minimum gain threshold.
- [x] Install CatBoost locally, smoke-test it, and compare the one predeclared native configuration with training-only early stopping and bounded threads. CatBoost qualified on both fixed seeds; HGB-1000 was also measured but missed the minimum gain on seed 2024. Results and exact metadata are in `artifacts/challengers/compare_metadata.json` and `approach.md`.
- [x] Run the authorized numeric-versus-native-categorical rating experiment. Only the 13 ratings changed representation; observed zeros remained valid, numeric delay NaNs remained numeric NaNs, and missing/non-integral/out-of-domain ratings were rejected. The preserved seed-42 result gained `+0.0000528012`; the explicit second-split mode trained only seed 2024 under the 85-minute internal/90-minute external bound and gained `-0.0000063016`. Both gains failed the required `+0.0002` gate, so the run was information-only and the current blend remained selected. The prior failed metadata/CSV were reused after provenance checks and remain byte-identical; artifacts are under `artifacts/categorical_ratings/` and `artifacts/categorical_ratings_seed2024/`.
- [x] Run the authorized fixed five-feature CatBoost/LightGBM 50/50 pass with `rating_mean`, `low_rating_count`, departure/arrival delay indicators, and arrival-delay missingness. The exact five-feature function preserved observed zero ratings and missing delay values, reused the recorded seed-42/2024 arrays and prior inner splits, and passed all integrity checks. The engineered blend was rejected because gains versus the current unengineered blend were `-0.0001212416` and `+0.0000129975`, both below the required `+0.0002` on each seed. The seed-31415 confirmation, refit, reload, and final export were skipped; artifacts are under `artifacts/feature_pass/`.
- [x] Assess and run the one authorized original-source augmentation experiment. The CC0 source was schema-checked, canonical feature-overlap filtered without IDs or labels, deduplicated with conflicting labels rejected, and reduced from 129,880 to 129,859 clean rows. Slow numeric-rating CatBoost plus the saved corrected LightGBM failed the fixed +0.0002 AUC gate on seeds 42 and 2024 (`-0.0000096073`, `-0.0002922921`); retain the current blend. No third seed, refit, export, upload, commit, or push was performed. Details and all hashes are in `approach.md` and `artifacts/source_augmentation/compare_metadata.json`.
- [x] Run the authorized fixed 50/50 CatBoost depth-6 + LightGBM blend confirmation on fresh stratified seed 31415, with both models sharing the same outer and inner partitions. Recompute prior two-seed evidence from saved predictions after unique-ID alignment. The fresh blend gained `+0.0003602265` AUC over CatBoost and strictly exceeded LightGBM; artifacts are under `artifacts/blend_50_50/`.
- [x] Select the challenger using local evidence and the predeclared two-seed rule. CatBoost qualified with AUC gains of 0.000549887 and 0.000533392 over the original HGB and was selected among qualifying challengers. The original model-selection policy and root `artifacts/` baseline remain unchanged; LightGBM, feature engineering, and ensembles were skipped.
- [x] Run the separately authorized single deeper CatBoost experiment with only `depth=8` changed from the existing depth-6 configuration. On seeds 42 and 2024, gains versus prior depth-6 CatBoost were `+0.0000541864` and `-0.0000598357`; the `0.0002`-each-seed gate was not met. Retain the prior depth-6 candidate and create no new final candidate. Exact results and provenance are in `artifacts/deeper_catboost/compare_metadata.json` and `approach.md`.
- [x] Run the initial bounded native LightGBM experiment with the predeclared 1500-tree configuration. Review found that the binary objective registered both AUC and `binary_logloss`; preserve its files unchanged in `artifacts/lightgbm_dual_metric_review/` as the invalid stopping-policy run.
- [x] Rerun the corrected LightGBM configuration with explicit `metric=auc`, recorded CatBoost inner splits, disjoint smoke subsets, and a 0.0002 AUC gain gate on both seeds. The authoritative run measured gains of `+0.0001394238` and `-0.0000482102`, so it was rejected; exact results and provenance are in `artifacts/lightgbm/compare_metadata.json` and `approach.md`.
- [x] Refit LightGBM on all rows at the median one-based best-iteration count `783` from prior counts `879,783` plus fresh count `778`, reuse the unchanged CatBoost final artifact, export the fixed 50/50 blend, and verify native-model reloads and CSV round-trip.
- [x] Run the authorized one longer-CatBoost experiment with only the iteration cap changed from `1200` to `3000`. Reuse the recorded outer/inner arrays and saved corrected LightGBM predictions, require a `0.0002` fixed-blend gain on both seeds, and preserve the current blend when the measured gains were `0.0000000000` and `+0.0001091262`. Fresh confirmation, refit, and final export were skipped; artifacts are under `artifacts/longer_catboost/`.
- [x] Run the authorized slower-learning pass: LightGBM `0.02`/`3000` trees, CatBoost `0.03`/`6000` iterations, patience `200`, everything else frozen on the recorded splits. All three fixed blends failed the `0.0002` both-seed gate (best `+0.0000803`/`+0.0001753`), so the current blend was retained and no fresh confirmation ran. Standalone slow CatBoost improved on both seeds; slow LightGBM did not. Artifacts are under `artifacts/slower_learning/`.
- [x] After explicit user authorization, export the below-gate slow-CatBoost candidate: all-row refit at the rounded median one-based count `4740`, reuse of the existing 783-tree LightGBM, and a validated 299,844-row `id,satisfaction` CSV. Fresh-process reloads reproduced the predictions and the CSV round-trip difference was `2.22e-16`. No Kaggle upload was performed.
- [x] Run the one fixed 75/25 slow-CatBoost/original-corrected-LightGBM validation blend. It scored `-0.0000775503` and `-0.0000096454` versus the current 50/50 blend on seeds 42 and 2024, failed the `+0.0002`-each-seed gate, and retained the current blend. Outputs are under `artifacts/blend_75_25/`; no model fit or export was performed.
- [x] Run exactly one fixed memory-bounded RandomForest configuration after the phase-1 commit/push. The standalone RF AUCs were `0.9550480356` and `0.9554150226`; the fixed 90/10 RF blend gains versus the current 50/50 blend were `-0.0000581150` and `-0.0000931299`, so the `+0.0002`-each-seed gate failed and the current blend was retained. Both seeds used 503,737 inner-fit rows, two threads, peak RSS below 1,020 MB, and unchanged data/pre-existing-artifact hashes. Outputs are under `artifacts/random_forest/`; the initial post-fit recording failure is preserved under `artifacts/random_forest_initial_failure/`.
- [x] Run exactly one fixed CatBoost seed-average experiment with only `random_seed=2025` changed from the existing slow CatBoost seed `42`. Reuse the exact outer and inner arrays, saved slow-CatBoost/LightGBM components, and fixed `.25/.25/.5` candidate math. The candidate gained `+0.0000334072` and `+0.0000413610` AUC on seeds `42` and `2024`, both below the predeclared `+0.0002` gate, so retain the current blend. The two fits completed in `1,472.386` seconds with best/actual counts `4365/4365` and `5134/5134`; output and integrity records are under `artifacts/catboost_seed_average/`. No third split, refit, export, upload, or commit/push was performed.
- [x] Run exactly one fixed Eco/business/loyal specialist experiment using the existing five folds and seed-42 inner splits. Fit only group rows for `Class=Eco`, `Type of Travel=Business travel`, and `Customer Type=Loyal Customer`; route only those rows through fixed `.75` baseline/`.25` specialist weights. The mean paired global AUC gain was `+0.0000341026` across five positive folds and pooled gain was `+0.0000343239`, below the `+0.0002` advancement rule; retain the current blend. Outputs and integrity records are under `artifacts/eco_business_specialist/`; no refit, export, upload, commit, or push was performed.
- [x] After explicit user authorization, run the minimal below-gate specialist export. Validate the completed five-fold/group/configuration/OOF provenance, derive the full-fit count from one-based metadata counts `[791,954,764,1182,1038]` to `954`, fit only the exact 65,542-row group, route the frozen baseline only for the matching test group, preserve outside-group CSV probability strings, and save only under `artifacts/eco_business_specialist_export/`. The 299,844-row export passed clean subprocess reload (`0.0` maximum difference), exact schema/order/probability checks, four-input hash checks, and all 115 prior-artifact hash checks. The CV gate remains false; this is user-authorized descriptive output, not a winner. No upload, commit, or push was performed.

### Optional higher-compute extension — before final selection

Run only after Phases 3–4 have measured results, Mac access is confirmed, execution/transfer is authorized, and time permits. Every task below is optional. If the gate is not met, record the skip reason and continue directly to Phase 6 with the VPS-selected model.

- [ ] Confirm actual Mac CPU, available memory/disk, native ARM64 Python, and package support. Create a separate local environment; do not copy the Linux virtual environment.
- [ ] Transfer data only through an authorized private route after checking data-access restrictions. Verify input SHA-256 fingerprints and preserve ignored data/artifact paths; never use Git to transfer CSVs.
- [ ] Reproduce baseline scores on identical split IDs and settings. Record platform/version differences and investigate material discrepancies before comparing new experiments.
- [ ] Benchmark one representative tree fit for runtime and peak memory. Set explicit wall-clock/trial budgets, concurrent-fit count, and per-fit thread limits; leave CPU/memory headroom rather than launching every trial at once.
- [x] Produce five-fold stratified splits shared across shortlisted models. Fit transforms and internal early stopping within each fold's training partition only. Verify each training row receives exactly one out-of-fold prediction and no validation row enters its own fit. Stratified folds used seed `2026`; every fold fitted CatBoost seeds `42`/`2025` and the original LightGBM. The fixed two-seed candidate gained `+0.0000091` mean paired AUC (4/5 folds positive, pooled OOF `+0.0000081`), below the `0.0002` rule, so the current blend was retained. Results are under `artifacts/five_fold_cv/`.
- [x] Save row-aligned out-of-fold predictions, per-fold/overall AUC and log loss, settings, runtime, and resource measurements. Compare candidates on the same folds; do not label fold variation a formal confidence interval. `artifacts/five_fold_cv/oof_predictions.npz` holds all `699,635` rows once, and the metadata records per-fold metrics, fit times, and resource snapshots.
- [x] Run the fixed subgroup specialist follow-up on the existing OOF baseline. Record group coverage, fold-local membership, full global paired metrics, subgroup calibration/error diagnostics, fixed routing, resources, and unchanged input/pre-existing-artifact hashes. Keep the result as development/stability evidence rather than independent confirmation.
- [ ] Compare CatBoost and then LightGBM under bounded, incremental configuration searches. Install each only when justified; stop when runtime limits or inconsistent gains make further trials unhelpful.
- [ ] Repeat promising improvements with another fold seed if the budget permits. Preserve the original holdout results as a reference, not a claim of independent evidence after repeated tuning.
- [ ] Measure held-out permutation importance and inspect adequately sized travel-type, cabin-class, loyalty, and age-band error slices. Record correlated-feature and noncausal interpretation limits.
- [ ] Reuse the existing conditional rating-encoding, feature-aggregate, and original-source-data tasks; do not create duplicate experiment pipelines or bypass their access/validation checks.
- [ ] Test a simple blend using aligned out-of-fold predictions from complementary models. Restrict weight search and confirm its gain on another seed or an untouched confirmation split; retain a single model if improvement is not repeatable.
- [ ] Optional learning experiment: confirm PyTorch MPS availability/operation support, smoke-test a small tabular network, and benchmark it against CPU and the tree baseline. Use training-only preprocessing/early stopping, cap training time, and skip if neither learning value nor predictive complementarity justifies it. Do not assume CatBoost/LightGBM use the Apple GPU.
- [x] Record measured results or explicit skip reasons in `approach.md`. Select the final method before Phase 6; Mac experiments were not made a delivery prerequisite.

Exit check: the selected method has confirmation beyond a single favorable holdout; optional work is either measured or explicitly skipped, with no promised leaderboard outcome.

## Phase 6 — Refit and generate a validated prediction file

- [x] Freeze the challenger feature definitions, preprocessing, hyperparameters, seed, and iteration count using local validation evidence only. CatBoost used median `best_iteration + 1`, rounded to 1186; the outer holdouts were not used as early-stopping evaluation data.
- [x] Refit the selected CatBoost model on all training rows with early stopping disabled. Save the native model under `artifacts/challengers/`; preserve the original root final model and submission.
- [x] Save challenger feature spec, model, submission, validation predictions, and metadata locally. A clean interpreter reloaded the native model and reproduced all 299,844 predictions exactly; CSV round-trip difference was `2.22e-16`.
- [x] Predict satisfaction probabilities for every test row in original test order. The export contains positive-class probabilities, not labels or negative-class probabilities.
- [x] Write `artifacts/challengers/submission.csv` with exactly `id,satisfaction` and no dataframe index column; preserve the original `artifacts/submission.csv`.
- [x] Reload the written CSV and verify 299,844 rows, exact test ID sequence, unique IDs, no missing/infinite predictions, and every probability in [0, 1].
- [x] Add/run a small synthetic export check that fails on malformed schema, duplicate/misaligned IDs, and out-of-range/nonfinite probabilities. Real-file checks also passed.
- [x] Verify the source CSV fingerprints remained unchanged and `git status` shows no raw data, fitted models, or predictions staged for publication.
- [x] Record the corrected rejected LightGBM result without creating a final LightGBM model or submission; retain the prior depth-6 CatBoost candidate and preserve the invalid first-run archive.

Exit check: a real, locally validated prediction artifact exists and its generating run is reproducible. No upload has occurred.

## Phase 7 — Validate implementation and hand off

- [x] Run quality gates in order: format, lint, typecheck, test. Use minimal project-local tooling/configuration where missing; keep checks focused on implemented source and tests, not source-data formatting. The seed-average source gates passed and the full regression suite passed 71 tests before the real run.
- [x] Rerun the runnable regression check and the documented baseline/challenger commands after the final code change. The follow-up wrapper regression checks are synthetic only: ordinary failure persistence, protected-artifact finalization, and completed-rejected/timeout CLI summaries. No depth-8 model rerun was needed; completed results remain preserved.
- [x] Run the final LightGBM source gates and one externally bounded corrected real experiment. The candidate completed and was rejected on both fixed seeds; no final LightGBM refit or submission was produced. The first dual-metric run remains archived and unchanged.
- [x] Run the blend source gates and the externally bounded fresh-split confirmation. The gate passed; final blend outputs and metadata are under `artifacts/blend_50_50/`. No standalone LightGBM submission, upload, commit, or push occurred.
- [x] After the completed blend run, add only wrapper guard regressions for prior-artifact provenance/configuration and input/protected-artifact failure persistence. No real rerun or artifact modification was performed after this guard-only change.
- [x] Review the final diff for scope, credentials, accidental data exposure, leakage, unused experimental code, and dependency additions.
- [x] Record exact reproduction commands including `compare.py`, environment versions, selected settings, measured validation results/runtime, protected baseline fingerprints, and challenger artifact paths. Existing documentation was updated; no separate design/report file was created.
- [x] Record the longer-CatBoost command, measured two-seed gains, rejected branches, artifact size, input/protected-artifact integrity, and the absence of a final submission in `approach.md`.
- [x] Apply independent-review wrapper guards for saved-ID alignment, absolute artifact-directory reuse, timeout cleanup, selected-iteration reuse, and post-run budget accounting. Preserve the completed run and all artifacts; do not rerun training.
- [x] Present results and remaining uncertainties to the user. Changes remain uncommitted; no push or publication occurred.
- [x] Run the authorized source augmentation experiment and record its rejection, exact counts, resource limits, integrity hashes, license/provenance, and skipped follow-up actions. No submission or publication occurred.
- [x] Run the authorized CatBoost seed-average experiment and record the positive below-gate gains, exact resources, integrity hashes, and skipped follow-up actions. No submission or publication occurred.
- [x] Record and review the authorized below-gate specialist export: model, feature/group/routing specs, metadata-derived 954-tree count, 65,542/28,227 train/test subgroup counts, actual runtime/resource observations, baseline/submission hashes, CSV text-preservation checks, clean subprocess reload, unchanged input hashes, and unchanged prior artifact hashes. Do not claim leaderboard improvement and do not upload the file.

- [x] Run the fixed LightGBM capacity diagnostic (`num_leaves=31 -> 63` only) on the original two holdouts. Blend gains were `+0.0000843972` / `+0.0000850509`, both below `+0.0002`; retain the current blend. The real run completed in `52.855` seconds within the 28/30-minute limits, with best/actual counts `375` / `517`, unchanged four-input and 120-prior-artifact hashes, and 85 passing tests. Outputs: `artifacts/lightgbm_capacity63/`. No extra folds, refit, export, upload, commit or push.

- [x] Run one fixed CPU-only XGBoost diagnostic with the original two holdouts, fit-only categorical mappings, inner AUC early stopping and `.8` current blend / `.2` XGBoost routing. Gains were `-0.0000053100` / `-0.0000177763`; retain the incumbent. The real run completed in `84.426` seconds, with one-based best counts `731` / `773`, unchanged four-input and 124-prior-artifact hashes, and 89 passing tests. Added only `xgboost-cpu==3.2.0` locally and to requirements. Outputs: `artifacts/xgboost_cpu/`; no refit, export, upload, commit or push.

- [x] Run the fixed source-only teacher feature experiment with overlap-cleaned source data, source-only teacher early stopping/full-source fit and original LightGBM31 students. The `.5` existing slow CatBoost / `.5` teacher-feature student blend gained `+0.0002335277` / `+0.0002644851`, passing the frozen two-holdout screening gate. Teacher: 531 rounds on 129,859 source rows; students: 545/313 rounds. Real run `70.903` seconds; four inputs and 127 prior artifacts unchanged; 92 tests pass. Outputs: `artifacts/source_teacher_feature/`. Stronger validation still needs approval; no final competition refit, export, upload, commit or push occurred.

- [x] Validate the fixed teacher-feature candidate on the existing five folds, fitting only five new LightGBM students and reusing the frozen teacher cache/CatBoost predictions. All five gains exceed `+0.0002`; mean paired gain `+0.0002492560`, pooled gain `+0.0002476338`. Frozen CV gate passes. Real run `123.471` seconds; all 699,635 OOF rows covered once; four inputs and 131 prior artifacts unchanged; 95 tests pass. Outputs: `artifacts/source_teacher_cv/`. These reused development folds are not an untouched test. No final refit, export, upload, commit or push.

- [x] After explicit approval, perform the qualified final teacher/student refit and local export. Derived fixed counts: source-only teacher 531, competition student 533; reused the unchanged 4,740-tree CatBoost. Saved `artifacts/source_teacher_export/submission_slowcat_source_teacher_50_50.csv` with SHA-256 `25fe7ff6b4b2d0c48d6b8fabc1c1de6d6d101898e8ed8b0ee2e52acab8200247`. All 299,844 test IDs/schema/probabilities and fresh-process native predictions verified; independent CSV difference 0.0. Four inputs and 138 prior artifacts unchanged; 104 tests pass; real run 57.118 seconds. No upload, commit or push.

- [x] Test the frozen teacher feature in CatBoost while keeping the winning teacher-feature LightGBM fixed. Compared against the new public-0.95845 configuration on the original holdouts. Blend gains were `-0.0000250528` / `-0.0001942532`; reject this fixed candidate and keep the incumbent. Best/actual tree counts `1532` / `1561`; actual run `571.536` seconds within the 43/45-minute limits. Four inputs and 145 prior artifacts unchanged; 110 tests pass. Outputs: `artifacts/catboost_teacher_feature/`. No CV, final refit, export, upload, commit or push.

- [x] Run the fixed 17-feature distance-group profile test in the winning teacher-LightGBM, with original CatBoost/teacher/settings/50-50 weights unchanged. Fit tables on inner model-fit predictors only. Gains versus the current winner: `+0.0013083434` / `+0.0011140500`; both pass the `+0.0002` screening gate. Actual run `64.020` seconds; best/native rounds `560` / `401`; independent table reconstruction, ID/blend/metric checks and all 118 tests pass. Four inputs and 148 prior artifacts unchanged. Outputs: `artifacts/distance_profile_feature/`. No CV, final refit, export, upload, commit or push.
- [x] Validate the fixed distance-profile student on the recorded five folds after separate approval. Refit profile tables only on each fold's inner model-fit predictors and reused frozen teacher/original CatBoost caches. Gains `+0.0013250688`, `+0.0011974558`, `+0.0013105411`, `+0.0011294748`, `+0.0012708839`; mean `+0.0012466849`, five positive folds, pooled `+0.0012488829` (`0.9591080056` to `0.9603568886`). Gate passes. Runtime `186.764` seconds; 699,635 OOF rows exactly once; 154 prior artifacts/four inputs unchanged; all 120 tests pass. Outputs `artifacts/distance_profile_cv/`; independent table/metric verification passed. Independent code review completed against unchanged source hashes: the timeout concern was resolved by checking the actual external launcher; focused re-review passed with no blockers. No final refit/export/upload/commit/push.
- [x] After user authorization, commit/push the existing checkpoint (`fdb111eb3cda5b1900c3a5274cb4a6b5da69e633`), then refit/export the qualified profile student at the recorded median `542` rounds. Full-training predictor-only profile table has `3474` groups; frozen teacher531/original CatBoost4740 and fixed 50/50 blend reused. Export verified against fresh-process native inference on all `299844` test rows, maximum difference `2.220446049250313e-16`; all 166 prior artifacts and four inputs unchanged. Export runtime `70.897` seconds. CSV `artifacts/distance_profile_export/submission_slowcat_teacher_distance_profile_50_50.csv`, SHA-256 `048abb59dc3ffb20e495dd414f7e7be3ca0e795d3fa66145516957c4b1d148e7`. Independent export-code re-review passed; all 127 tests pass. The same user request explicitly authorized the subsequent upload and final commit/push; previous submitted files remain preserved.

- [x] Run the approved fixed source-neighbour probability screen against the public-0.95982 configuration. One source-only exact-distance Hamming-weighted feature; unchanged teacher/profile/LightGBM/CatBoost/50-50 weights. Gains `+0.0000626681` / `+0.0001052498` are positive but both below the `+0.0002`-each-seed gate; retain the incumbent, with no neighbour CV/refit/export/upload. Real run `95.506` seconds; oracle verified 4,487 rows, all 171 prior artifacts/four inputs unchanged, independent review and 137 tests pass. Outputs `artifacts/source_neighbor_feature/`. New source/tests and docs were saved; the user subsequently authorized their checkpoint commit/push.

Exit check: working code and prediction artifact are backed by actual execution, with no unsupported success claims.

## Phase 8 — Submit only with explicit approval

- [x] Record that the agent made no new submission for the deeper experiment; the user manually submitted prior models. The deeper candidate did not qualify and no new final CSV exists.
- [x] Record that the fixed 50/50 blend was exported locally after passing its predeclared gate; the agent made no new Kaggle submission.
- [ ] Recheck live rules, deadline, participation eligibility/acceptance status, and available submission allowance. Resolve outstanding compliance questions before upload.
- [x] Obtain explicit approval for the exact source-teacher prediction CSV and submission action. The user separately authorized uploading `submission_slowcat_source_teacher_50_50.csv`; its SHA-256 and schema were rechecked before submission.
- [x] Upload the approved source-teacher CSV once and read back Kaggle submission `57698563`: `COMPLETE`, public ROC AUC `0.95845`. The previous `0.95831` submission and local file were preserved.
- [x] Record the API-verified public ROC AUC `0.95845` (`+0.00014`) separately from local CV results. Competition-list API rank was `1968` after scoring, versus `2173` immediately before upload; ranks are time-dependent. Final private score and final-submission selection remain unverified.
- [x] Upload the user-authorized distance-profile CSV once after hash/schema/deadline/entered-status and duplicate checks. Read back submission `57726880`: `COMPLETE`, public ROC AUC `0.95982`, actual improvement `+0.00137` over `0.95845`. API rank observed `2027` before and `1652` after scoring (`3323` teams); nine daily submissions remained immediately after upload. The old CSVs remain unchanged; final/private results and final-submission selection are not established.
- [ ] Confirm final-submission selections with the user before the deadline; record the actual selections after verification.
- [ ] If a prize/code-verification request occurs, provide reproducible training code, inference code, environment details, and documentation as required. Do not create a public repository or publish competition code merely to meet an ordinary prediction submission.

Exit check: approved submissions/selections are verified on Kaggle, or this phase remains pending with an explicit reason.
