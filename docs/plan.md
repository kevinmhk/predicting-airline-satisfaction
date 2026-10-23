# Phased execution plan

Strategy and evidence: [approach.md](approach.md). Official snapshots: [overview.md](overview.md), [data.md](data.md), and [rules.md](rules.md).

`[x]` means verified complete. `[ ]` means not completed. Optional phases may be skipped with a short reason; do not mark an unrun experiment complete. This is a task list, not authorization to execute it. No model scores exist yet.

## Phase 0 — Establish facts and boundaries

- [x] Create the local project, Git repository, and public-documentation snapshots.
- [x] Download the official competition files and verify their move into `data/` with before/after SHA-256 checks.
- [x] Restore `/data/` to `.gitignore` and remove the uploaded data commit from branch history; verify a fresh remote clone contains no data history. GitHub caches and existing clones are outside that guarantee.
- [x] Read all saved documentation and inspect train, test, and sample-submission files locally.
- [x] Verify schema compatibility, unique/disjoint train/test IDs, and sample-submission ID order.
- [x] Measure target balance, missingness, feature cardinalities, basic distribution differences, and exact duplicate/overlap hashes.
- [x] Record observations, model proposals, limitations, and compliance questions in `approach.md`.
- [ ] Confirm the user's Kaggle account, competition participation status, and unresolved eligibility/employer/sanctions checks before entry or submission. Do not infer acceptance from a successful download.
- [ ] Resolve external-AI data/code sharing against the competition rules before providing competition material to hosted coding/training services. Keep local analysis separate from that decision.
- [ ] Obtain authorization for the first implementation increment. Do not register, accept terms, spend, publish, commit, push, or submit as an implicit part of implementation.

Exit check: observed findings are distinguished from hypotheses, and the next execution scope and access boundaries are explicit.

## Phase 1 — Prepare the smallest reproducible environment

- [ ] Inspect current repository instructions and working-tree state before editing.
- [ ] Create a project-local virtual environment; do not modify the unrelated RD-Agent environment used for inspection.
- [ ] Install and pin pandas, NumPy, and scikit-learn. Confirm the installed scikit-learn API supports the selected categorical boosting path.
- [ ] Set bounded CPU thread counts, initially no more than two; check available memory and disk before training.
- [ ] Extend `.gitignore` for the virtual environment, Python caches, and a single local `artifacts/` directory while preserving `/data/`.
- [ ] Define one training entry point and one lightweight test module. Avoid package scaffolding, configuration frameworks, and orchestration services.
- [ ] Store future models, metrics, validation predictions, and submission files under `artifacts/`, not `docs/` or tracked source folders.

Exit check: the isolated environment imports required packages, dependency versions are recorded, and Git excludes data/runtime artifacts.

## Phase 2 — Define and test the validation contract

- [ ] Implement input checks for required fields, train/test predictor agreement, valid binary target, and unique/disjoint IDs. Return an actionable error for invalid input.
- [ ] Exclude `id` and `satisfaction` from predictor inputs. Separate nominal categorical features from numeric features explicitly.
- [ ] Create the stratified 80/20 outer split with seed 42. Record split row IDs or indices locally so models use identical validation rows.
- [ ] Check split disjointness, complete training-row coverage, and presence of both classes in each partition.
- [ ] Record SHA-256 input fingerprints and exact dependency versions with each run. Leave source data unchanged.
- [ ] Define common evaluation output: model/settings, split seed, ROC AUC, log loss, elapsed fit time, and convergence/early-stopping information.
- [ ] Write a small synthetic regression check before implementing nontrivial preprocessing/export behavior. Cover missing delay, an unseen category, and exclusion of ID/target features without using raw competition rows in test fixtures.
- [ ] Check that preprocessing is trained only on the fit partition. Reserve the outer validation set for evaluation, not early stopping or transformation fitting.

Exit check: the synthetic check passes, split membership is reproducible, and both candidate models share the same evaluation contract.

## Phase 3 — Measure the baseline

- [ ] Compute a constant-probability reference from fit-partition target prevalence; evaluate its ROC AUC and log loss.
- [ ] Build the logistic regression pipeline with training-fitted median imputation, numeric scaling, and one-hot categorical encoding with unknown-category handling.
- [ ] Run a small smoke fit to verify valid two-class probabilities and the end-to-end metric path before a full baseline fit.
- [ ] Train on the full fit partition. Check convergence warnings; adjust iteration limits or solver only if needed.
- [ ] Save outer-validation probabilities with row IDs, metrics, settings, seed, input fingerprints, and elapsed time locally.
- [ ] Inspect AUC/log-loss results against the constant reference. Investigate implausible results or failed convergence before advancing; do not invent a target AUC.
- [ ] Record measured results and limitations in `approach.md` without replacing the distinction between observations and proposals.

Exit check: a reproducible baseline has real validation scores and no unresolved fitting or probability-validity failures.

## Phase 4 — Compare histogram gradient boosting

- [ ] Build training-fitted category mappings, safe unseen-category handling, and explicit categorical-column markers for histogram boosting. Preserve NaN numeric delays.
- [ ] Extend the synthetic check to cover the boosting transformation and stable feature order. Verify missing/unseen values do not cause an invalid prediction.
- [ ] Set a small, bounded initial configuration. Use only an internal subset of the fit partition for early stopping.
- [ ] Run a smoke fit, then train the first full histogram-boosting contender on the same outer split as the baseline.
- [ ] Save aligned validation probabilities, settings, actual iterations, metrics, and runtime under `artifacts/`.
- [ ] Compare ROC AUC, log loss, runtime, and observed resource usage against logistic regression. Check major error slices by class/travel type/customer type where counts support interpretation; report associations, not causal conclusions.
- [ ] Keep the simpler baseline if boosting gives no useful benefit; document the measured reason rather than expanding the search automatically.

Exit check: both models have directly comparable measured results, with a provisional choice and explicit uncertainty.

## Phase 5 — Confirm the choice and run only justified experiments

- [ ] Re-evaluate shortlisted candidates on another fixed stratified split seed or three-fold stratified validation. Keep the comparison protocol identical across candidates.
- [ ] Check whether model ranking and claimed improvements are repeatable. Treat tiny or inconsistent differences as uncertain.
- [ ] Optional: install CatBoost locally, smoke-test it, and compare a small fixed configuration set with training-only early stopping and bounded threads. Skip if the measured baseline is sufficient or time/resources do not justify it.
- [ ] Optional: compare rating features as numeric versus categorical, including treatment of zeros as real observed values rather than assumed missing values.
- [ ] Optional: compare a small set of low-rating counts, rating averages, or delay indicators. Keep only repeatable gains; remove unused experimental code.
- [ ] Optional: assess whether original-source data is worth a separate experiment. Obtain download authorization and check terms, schema, overlap, and distribution before using it. Do not infer test labels or hand-label records.
- [ ] Optional: evaluate a simple blend using validation predictions aligned by row ID. Keep it only if it beats the best component consistently; skip stacking infrastructure.
- [ ] Select the final model/settings using local evidence and available time. Record why discarded alternatives were not kept.

Exit check: the selected method has confirmation beyond a single favorable holdout; optional work is either measured or explicitly skipped, with no promised leaderboard outcome.

## Phase 6 — Refit and generate a validated prediction file

- [ ] Freeze final feature definitions, preprocessing, hyperparameters, seed, and iteration count using local validation evidence only.
- [ ] Refit the chosen preprocessing/model on all training rows. If early stopping was used, select a fixed full-data iteration policy from validation evidence rather than holding out the outer validation data again or inspecting test labels.
- [ ] Save the fitted pipeline/model and run metadata locally. Verify reload produces equivalent probabilities on a small local check.
- [ ] Predict satisfaction probabilities for every test row in original test order. Do not export class labels or the negative-class probability by mistake.
- [ ] Write `artifacts/submission.csv` with exactly `id,satisfaction` and no dataframe index column.
- [ ] Reload the written CSV and verify 299,844 rows, exact test ID sequence, unique IDs, no missing/infinite predictions, and every probability in [0, 1].
- [ ] Add/run a small synthetic export check that fails on malformed schema, duplicate/misaligned IDs, and out-of-range/nonfinite probabilities. Do not replace the real-file checks with synthetic checks alone.
- [ ] Verify the source CSV fingerprints remain unchanged and `git status` shows no raw data, fitted models, or predictions staged for publication.

Exit check: a real, locally validated prediction artifact exists and its generating run is reproducible. No upload has occurred.

## Phase 7 — Validate implementation and hand off

- [ ] Run quality gates in order: format, lint, typecheck, test. Use minimal project-local tooling/configuration where missing; keep checks focused on implemented source and tests, not source-data formatting.
- [ ] Rerun the runnable regression check and the documented training/export command after the final code change. Report real outputs and any unrun/failed step explicitly.
- [ ] Review the final diff for scope, credentials, accidental data exposure, leakage, unused experimental code, and dependency additions.
- [ ] Record exact reproduction commands, environment versions, selected settings, measured validation results/runtime, and the prediction artifact path. Use existing documentation rather than creating a separate design/report document.
- [ ] Present results and remaining uncertainties to the user. Commit and push only if separately requested, with ignored data/artifacts excluded.

Exit check: working code and prediction artifact are backed by actual execution, with no unsupported success claims.

## Phase 8 — Submit only with explicit approval

- [ ] Recheck live rules, deadline, participation eligibility/acceptance status, and available submission allowance. Resolve outstanding compliance questions before upload.
- [ ] Obtain explicit approval for the exact prediction file and submission action. A completed local plan is not submission authorization.
- [ ] Upload the approved CSV through the authorized Kaggle account and read back submission status/score. Do not claim success from an upload response alone.
- [ ] Record the actual leaderboard result separately from local validation scores. Do not repeatedly tune to the public leaderboard at the expense of validation.
- [ ] Confirm final-submission selections with the user before the deadline; record the actual selections after verification.
- [ ] If a prize/code-verification request occurs, provide reproducible training code, inference code, environment details, and documentation as required. Do not create a public repository or publish competition code merely to meet an ordinary prediction submission.

Exit check: approved submissions/selections are verified on Kaggle, or this phase remains pending with an explicit reason.
