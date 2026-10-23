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
- [x] Select HGB because its gain repeated on both fixed splits; no complexity expansion was needed.

Exit check: both models have directly comparable measured results, with a provisional choice and explicit uncertainty.

## Phase 5 — Confirm the choice and run only justified experiments

- [x] Re-evaluate shortlisted candidates on another fixed stratified split seed (2024). Keep the comparison protocol identical across candidates.
- [x] Check whether model ranking and claimed improvements are repeatable. HGB won both fixed splits by more than the minimum gain threshold.
- [ ] Optional: install CatBoost locally, smoke-test it, and compare a small fixed configuration set with training-only early stopping and bounded threads. Skip if the measured baseline is sufficient or time/resources do not justify it.
- [ ] Optional: compare rating features as numeric versus categorical, including treatment of zeros as real observed values rather than assumed missing values.
- [ ] Optional: compare a small set of low-rating counts, rating averages, or delay indicators. Keep only repeatable gains; remove unused experimental code.
- [ ] Optional: assess whether original-source data is worth a separate experiment. Obtain download authorization and check terms, schema, overlap, and distribution before using it. Do not infer test labels or hand-label records.
- [ ] Optional: evaluate a simple blend using validation predictions aligned by row ID. Keep it only if it beats the best component consistently; skip stacking infrastructure.
- [x] Select the final model/settings using local evidence and available time. CatBoost, LightGBM, feature engineering, and ensembles were skipped because the bounded HGB result was sufficient for this delivery.

### Optional higher-compute extension — before final selection

Run only after Phases 3–4 have measured results, Mac access is confirmed, execution/transfer is authorized, and time permits. Every task below is optional. If the gate is not met, record the skip reason and continue directly to Phase 6 with the VPS-selected model.

- [ ] Confirm actual Mac CPU, available memory/disk, native ARM64 Python, and package support. Create a separate local environment; do not copy the Linux virtual environment.
- [ ] Transfer data only through an authorized private route after checking data-access restrictions. Verify input SHA-256 fingerprints and preserve ignored data/artifact paths; never use Git to transfer CSVs.
- [ ] Reproduce baseline scores on identical split IDs and settings. Record platform/version differences and investigate material discrepancies before comparing new experiments.
- [ ] Benchmark one representative tree fit for runtime and peak memory. Set explicit wall-clock/trial budgets, concurrent-fit count, and per-fit thread limits; leave CPU/memory headroom rather than launching every trial at once.
- [ ] Produce five-fold stratified splits shared across shortlisted models. Fit transforms and internal early stopping within each fold's training partition only. Verify each training row receives exactly one out-of-fold prediction and no validation row enters its own fit.
- [ ] Save row-aligned out-of-fold predictions, per-fold/overall AUC and log loss, settings, runtime, and resource measurements. Compare candidates on the same folds; do not label fold variation a formal confidence interval.
- [ ] Compare CatBoost and then LightGBM under bounded, incremental configuration searches. Install each only when justified; stop when runtime limits or inconsistent gains make further trials unhelpful.
- [ ] Repeat promising improvements with another fold seed if the budget permits. Preserve the original holdout results as a reference, not a claim of independent evidence after repeated tuning.
- [ ] Measure held-out permutation importance and inspect adequately sized travel-type, cabin-class, loyalty, and age-band error slices. Record correlated-feature and noncausal interpretation limits.
- [ ] Reuse the existing conditional rating-encoding, feature-aggregate, and original-source-data tasks; do not create duplicate experiment pipelines or bypass their access/validation checks.
- [ ] Test a simple blend using aligned out-of-fold predictions from complementary models. Restrict weight search and confirm its gain on another seed or an untouched confirmation split; retain a single model if improvement is not repeatable.
- [ ] Optional learning experiment: confirm PyTorch MPS availability/operation support, smoke-test a small tabular network, and benchmark it against CPU and the tree baseline. Use training-only preprocessing/early stopping, cap training time, and skip if neither learning value nor predictive complementarity justifies it. Do not assume CatBoost/LightGBM use the Apple GPU.
- [x] Record measured results or explicit skip reasons in `approach.md`. Select the final method before Phase 6; Mac experiments were not made a delivery prerequisite.

Exit check: the selected method has confirmation beyond a single favorable holdout; optional work is either measured or explicitly skipped, with no promised leaderboard outcome.

## Phase 6 — Refit and generate a validated prediction file

- [x] Freeze final feature definitions, preprocessing, hyperparameters, seed, and iteration count using local validation evidence only.
- [x] Refit the chosen preprocessing/model on all training rows. The fixed 200-iteration policy came from the two validation fits; the outer validation data was not reused for early stopping.
- [x] Save the fitted pipeline/model and run metadata locally. A clean interpreter reloaded the artifact successfully; the synthetic probability comparison was exact and the full submission comparison stayed within floating-point CSV round-trip precision.
- [x] Predict satisfaction probabilities for every test row in original test order. The export contains positive-class probabilities, not labels or negative-class probabilities.
- [x] Write `artifacts/submission.csv` with exactly `id,satisfaction` and no dataframe index column.
- [x] Reload the written CSV and verify 299,844 rows, exact test ID sequence, unique IDs, no missing/infinite predictions, and every probability in [0, 1].
- [x] Add/run a small synthetic export check that fails on malformed schema, duplicate/misaligned IDs, and out-of-range/nonfinite probabilities. Real-file checks also passed.
- [x] Verify the source CSV fingerprints remained unchanged and `git status` shows no raw data, fitted models, or predictions staged for publication.

Exit check: a real, locally validated prediction artifact exists and its generating run is reproducible. No upload has occurred.

## Phase 7 — Validate implementation and hand off

- [x] Run quality gates in order: format, lint, typecheck, test. Use minimal project-local tooling/configuration where missing; keep checks focused on implemented source and tests, not source-data formatting.
- [x] Rerun the runnable regression check and the documented training/export command after the final code change. The final run passed and its real outputs are recorded in `approach.md`.
- [x] Review the final diff for scope, credentials, accidental data exposure, leakage, unused experimental code, and dependency additions.
- [x] Record exact reproduction commands, environment versions, selected settings, measured validation results/runtime, and the prediction artifact path. Existing documentation was updated; no separate design/report file was created.
- [x] Present results and remaining uncertainties to the user. Changes remain uncommitted; no push or publication occurred.

Exit check: working code and prediction artifact are backed by actual execution, with no unsupported success claims.

## Phase 8 — Submit only with explicit approval

- [ ] Recheck live rules, deadline, participation eligibility/acceptance status, and available submission allowance. Resolve outstanding compliance questions before upload.
- [ ] Obtain explicit approval for the exact prediction file and submission action. A completed local plan is not submission authorization.
- [ ] Upload the approved CSV through the authorized Kaggle account and read back submission status/score. Do not claim success from an upload response alone.
- [ ] Record the actual leaderboard result separately from local validation scores. Do not repeatedly tune to the public leaderboard at the expense of validation.
- [ ] Confirm final-submission selections with the user before the deadline; record the actual selections after verification.
- [ ] If a prize/code-verification request occurs, provide reproducible training code, inference code, environment details, and documentation as required. Do not create a public repository or publish competition code merely to meet an ordinary prediction submission.

Exit check: approved submissions/selections are verified on Kaggle, or this phase remains pending with an explicit reason.
