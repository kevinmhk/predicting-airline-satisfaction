# Modeling approach

## Status and objective

The VPS-only implementation and local training/export run completed on 2026-10-23. A bounded challenger experiment completed locally on 2026-10-24 HKT (2026-10-23 UTC). Real validation results and artifact paths are recorded below. No competition submission was uploaded.

Predict the probability of passenger satisfaction for each test row. Optimize ROC AUC. Start with a reproducible CPU baseline, then compare boosted trees on identical validation rows. Execution tasks are in [plan.md](plan.md); no separate architecture document is needed.

## Sources and competition constraints

- [Overview snapshot](overview.md): objective, ROC AUC, prediction CSV format, timeline, and prizes.
- [Data snapshot](data.md): synthetic data inspired by the linked original airline satisfaction dataset; similar distributions are not guaranteed to be identical.
- [Rules snapshot](rules.md): participation, external tools, sharing, and winner obligations.
- These public-page snapshots were retrieved on 2026-10-23. Their extraction notes describe that retrieval, not subsequent data-download activity. Live rules and deadlines must be checked before entry or submission.
- The recorded final deadline is 2026-10-31 23:59 UTC / 2026-11-01 07:59 HKT.
- Ordinary entry requires prediction results in a CSV with header `id,satisfaction`, not a GitHub link or code upload. Provide probabilities, not thresholded labels.
- The recorded limits are 10 submissions per day, two selected final submissions, and three people per team. Private-leaderboard performance determines final ranking.
- Prize winners must deliver reproducible training/inference code, environment details, and documentation under Rules section 2.8, despite the winner-license field saying None.
- External data/models/tools are conditionally allowed. Data-access restrictions and private code-sharing rules still apply. The user confirmed that external AI assistance is permitted for this local implementation/training increment. This is user-confirmed, not independently host-verified. Hosted AI coding assistance was used to implement and review the code; raw CSV files were not uploaded to a public repository or external training service.
- Keep raw data local and ignored by Git. No registration, terms acceptance, spending, public sharing, or submission is authorized by this plan. Each needs separate user authorization; unresolved eligibility/compliance checks remain open.

## Executed VPS-only run

The following local commands were run in the project directory. The RD-Agent environment was not modified.

```text
/home/kevinmhk/miniforge3/envs/rdagent/bin/python -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-dev.txt
.venv/bin/ruff format train.py test_train.py
.venv/bin/ruff check train.py test_train.py
.venv/bin/mypy train.py test_train.py
.venv/bin/python -m unittest -v test_train.py
.venv/bin/python train.py --data-dir data --artifacts-dir artifacts
```

The final run used Python 3.10.21, pandas 2.2.3, NumPy 1.26.4, scikit-learn 1.5.2, SciPy 1.14.1, and joblib 1.4.2. BLAS/OpenMP-related thread limits were set to two. The recorded training entry-point runtime was 46.537 seconds; peak memory was not remeasured in this rerun.

Both candidates used the same stratified 80/20 outer split (seed 42). HGB early stopping used only its internal split from the outer fit partition. A second fixed stratified split (seed 2024) showed the same ranking. These holdouts overlap: agreement shows split-seed stability, not statistically independent confirmation. Values below are local validation results, not leaderboard scores.

| Candidate | Outer ROC AUC | Outer log loss | Confirmation ROC AUC | Confirmation log loss | Fit seconds (outer / confirmation) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant fit prevalence | 0.500000 | 0.686766 | not run | not run | 0 / not run |
| Logistic regression | 0.920455 | 0.346764 | 0.920458 | 0.346452 | 2.887 / 2.986 |
| HistGradientBoostingClassifier | 0.956800 | 0.232302 | 0.957212 | 0.230954 | 9.630 / 9.810 |

The initial selected model was histogram gradient boosting. Candidate fits used `max_iter=200`, `max_leaf_nodes=31`, learning rate `0.08`, `l2_regularization=0.1`, internal early stopping, `validation_fraction=0.1`, `n_iter_no_change=20`, `random_state=42`, explicit categorical feature markers, and two-thread limits. Both measured HGB fits reached 200 iterations. The logistic fits used 22 actual iterations. The HGB full-data refit used the median observed iteration count (200) with `early_stopping=false` and `validation_fraction=null`; its final settings retained `l2_regularization=0.1` and `random_state=42` without outer holdout reuse. Selection required at least `0.0005` ROC AUC gain on both fixed splits. Final fit time was 10.470 seconds; final elapsed time including inference, serialization, reload, and export validation was 13.917 seconds.

The fitted pipeline uses scikit-learn `ColumnTransformer` with numeric passthrough and `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=NaN, encoded_missing_value=NaN)` for the four categorical columns. A synthetic subprocess regression check loaded the joblib in a clean interpreter and reproduced probabilities exactly. The full exported artifact also loaded in a clean interpreter and reproduced all 299,844 submission probabilities within `1.11e-16` maximum absolute difference after CSV round-tripping; the in-process artifact check was exactly 0.0.

The HGB outer-holdout slice diagnostics were: ROC AUC 0.952254 for Business travel and 0.819473 for Personal Travel; 0.956935 for Loyal Customer and 0.919849 for disloyal Customer; and 0.932018, 0.897640, and 0.920502 for Business, Eco, and Eco Plus. These are associations within the holdout and are not causal findings.

Input SHA-256 fingerprints were unchanged before and after training: `train.csv` `2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36`; `test.csv` `ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6`.

The local artifacts are `artifacts/run_metadata.json`, `artifacts/final_model.joblib`, `artifacts/validation_logistic.csv`, `artifacts/validation_hist_gradient_boosting.csv`, split files, and `artifacts/submission.csv`. The submission file was reloaded and verified to contain exactly 299,844 rows, columns `id,satisfaction`, unique IDs in exact test order, finite probabilities in `[0, 1]`, and no index column. Synthetic negative checks cover bad schema, duplicate/misaligned IDs, and NaN/infinite/out-of-range probabilities. It remains local and ignored; submission authorization is still pending.

The initial run skipped CatBoost, LightGBM, manual feature engineering, ensembles, Mac/MPS work, and leaderboard iteration. The later CatBoost challenger below was separately authorized and run locally. No remote data, leaderboard call, submission, or public artifact was used.

## Bounded challenger experiment

The user authorized one bounded challenger increment after the initial HGB result. The entry point is [compare.py](../compare.py); it reuses `train.py` input validation, feature definitions, recorded `outer_split.npz` and `confirmation_split.npz`, probability checks, and submission validation. It does not modify the original artifacts in `artifacts/`.

The experiment used sequential fits on two CPUs with thread/OpenMP/BLAS limits of two. Smoke fits ran before the four full validation fits. The predeclared configurations were exactly:

- HGB-1000: the existing HGB preprocessing and hyperparameters, changing only `max_iter` from 200 to 1000; existing internal early stopping remained enabled with validation fraction 0.1 and patience 20.
- CatBoost native: `depth=6`, `iterations=1200`, `learning_rate=0.06`, `loss_function=Logloss`, `eval_metric=AUC`, `l2_leaf_reg=3`, `random_seed=42`, `thread_count=2`, `allow_writing_files=false`, and `early_stopping_rounds=100`, with the four nominal columns passed as native categorical features. Numeric NaN values remained native NaNs.

CatBoost 1.2.8 was installed only in `.venv` with `--no-cache-dir` and pinned in `requirements.txt`. The full comparison took 660.079 seconds. Smoke fits completed in 0.155 seconds for HGB-1000 (63 iterations) and 1.070 seconds for CatBoost (188 actual iterations, best iteration index 187). The process-level resident-memory high-water mark reached 1490.3 MB; this is not an isolated per-model memory measurement. CPU count and configured thread limit were both two.

| Candidate | Seed 42 AUC / log loss | Seed 2024 AUC / log loss | AUC gain over original HGB | Actual iterations | Fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original HGB-200 | 0.956799979 / 0.232302 | 0.957212140 / 0.230954 | reference | 200 / 200 | 9.630 / 9.810 |
| HGB-1000 | 0.957054352 / 0.231626 | 0.957403105 / 0.230237 | +0.000254373 / +0.000190965 | 361 / 499 | 17.114 / 21.658 |
| CatBoost native | 0.957349866 / 0.231259 | 0.957745532 / 0.230124 | +0.000549887 / +0.000533392 | 1171 / 1200 | 192.598 / 181.161 |

Timing limitation: the comparison metadata's `fit_seconds` includes prediction/evaluation, unlike the baseline's fit-only timer. Final-refit timing also includes export and reload checks. Do not treat these as like-for-like training-speed benchmarks.

The minimum qualifying gain was predeclared as 0.0002 AUC on both seeded holdouts. HGB-1000 did not qualify because its seed-2024 gain was 0.000190965. CatBoost qualified on both seeds and had the highest mean AUC, so it was selected as a challenger-only final candidate. The seeded holdouts overlap; this is a stability diagnostic, not independent confirmation.

CatBoost best iteration indices were 1170 and 1199. The all-training-row refit therefore used the median of `best_iteration + 1`, rounded from 1185.5 to 1186, with early stopping disabled. It produced [artifacts/challengers/final_model_catboost.cbm](../artifacts/challengers/final_model_catboost.cbm), [artifacts/challengers/feature_spec.json](../artifacts/challengers/feature_spec.json), and [artifacts/challengers/submission.csv](../artifacts/challengers/submission.csv). A clean process reloaded the native model and predicted all 299,844 test rows: maximum model-reload difference was 0.0 and maximum CSV round-trip difference was `2.22e-16`. The original `artifacts/submission.csv`, `final_model.joblib`, validation files, split files, and `run_metadata.json` remain the active baseline and were not replaced.

Validation predictions for both candidates and both seeds are saved under `artifacts/challengers/validation_*.csv`. Exact CatBoost internal 90/10 fit/validation row indices and corresponding IDs are recorded in `artifacts/challengers/compare_metadata.json`; each internal validation subset is disjoint from its outer holdout. The same metadata records settings, actual/best iterations, AUC, log loss, fit/runtime, resource counts, package versions, input hashes, and the selection decision.

The input fingerprints remained `train.csv` `2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36` and `test.csv` `ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6`. Protected baseline artifact fingerprints matched before and after the experiment; the recorded after-check is in `artifacts/challengers/protected_baseline_sha256.json`.

The user reported a public screenshot showing baseline AUC `0.95737`, rank `2427`, and top AUC `0.96387`. This is user-reported only, was not independently fetched, was not treated as a local metric, and was not used for selection. No local AUC leaderboard was called.

The final challenger gates and real path were run with:

```text
.venv/bin/ruff format train.py test_train.py compare.py
.venv/bin/ruff check train.py test_train.py compare.py
.venv/bin/mypy train.py test_train.py compare.py
.venv/bin/python -m unittest -v test_train.py
/usr/bin/timeout --signal=TERM --kill-after=30s 1200s .venv/bin/python compare.py --data-dir data --artifacts-dir artifacts
```

## Deeper CatBoost experiment

On 2026-10-24 HKT, the user authorized one deeper CatBoost experiment. The new
entry point was run as `compare.py --deeper-catboost` with a 1500-second hard
timeout. It ran only the new depth-8 CatBoost configuration; it did not rerun
logistic regression, HGB, or the existing depth-6 CatBoost fit. Numerical
thread limits were set before importing NumPy, pandas, and CatBoost. The exact
configuration was `depth=8`, `iterations=1200`, `learning_rate=0.06`,
`l2_leaf_reg=3`, `loss_function=Logloss`, `eval_metric=AUC`, `random_seed=42`,
`thread_count=2`, `allow_writing_files=false`, and
`early_stopping_rounds=100`. Only depth changed from the prior depth-6 run.

The run reused the recorded outer split arrays for seeds 42 and 2024 and the
exact prior CatBoost internal 90/10 seed-42 subsets. Outer holdouts were never
passed as `eval_set`. Fit-only seconds were captured immediately after each
CatBoost fit; process high-water RSS and thread-pool observations are recorded
in the metadata. Both validation fits completed:

| Split seed | Depth-6 AUC | Depth-8 AUC | Gain | Actual / best iteration | Fit seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9573498661 | 0.9574040525 | +0.0000541864 | 512 / 511 | 119.091 |
| 2024 | 0.9577455323 | 0.9576856966 | -0.0000598357 | 654 / 653 | 140.711 |

The required gain was at least `0.0002` on each seed. The depth-8 candidate was
rejected and the prior depth-6 CatBoost candidate was retained. Because the
candidate did not qualify, no all-row depth-8 refit, native depth-8 model, or
descriptive final submission was created. The completed validation outputs and
full provenance are under `artifacts/deeper_catboost/`:

- `compare_metadata.json` contains the exact one new config, source/package
  versions, input hashes before/after, complete split and internal indices/IDs,
  validation metrics, fit times, resource observations, and the rejection.
- `validation_catboost_depth8_seed42.csv` and
  `validation_catboost_depth8_seed2024.csv` contain only validation outputs.
- `protected_baseline_sha256.json` records the existing artifact fingerprints.

The raw input fingerprints were unchanged:
`train.csv` `2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36`;
`test.csv` `ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6`.
Every pre-existing file under `artifacts/` and `artifacts/challengers/` matched
its before/after SHA-256 fingerprint, including the prior native model,
submission, and metadata. No existing candidate was promoted or modified.

The user separately reported a prior public CatBoost result of AUC `0.95797`
and rank `2225`, improving on a reported AUC `0.95737` and rank `2427`. These
are user-reported public results only, were not live-verified, and were not
used for local selection. The agent made no new submission; the user manually
submitted prior models. Submission/publication, commit, push, and Phase 8
remain pending.

After the final source change, the required gates passed in order:

```text
.venv/bin/ruff format compare.py train.py test_train.py
.venv/bin/ruff check compare.py train.py test_train.py
.venv/bin/mypy compare.py train.py test_train.py
.venv/bin/python -m unittest -v test_train.py
```

The follow-up orchestration review is wrapper-only. Synthetic tests cover
ordinary fit failure persistence, input/protected-artifact finalization, and the
completed-rejected and timeout CLI summaries. The completed depth-8 rejection
artifacts above were not rerun or modified; no model training is required for
these checks.

## Observed local data

All three downloaded CSVs were inspected locally. Only aggregate statistics were returned to the conversation.

- `train.csv`: 699,635 rows and 23 columns: `id`, 21 predictors, and `satisfaction`.
- `test.csv`: 299,844 rows and 22 columns: `id` and the same 21 predictors.
- `sample_submission.csv`: 299,844 rows and two columns. Its IDs match test IDs in exactly the same order.
- Target values are boolean: 310,339 satisfied and 389,296 not satisfied; positive prevalence is 44.36%. No class rebalancing is justified initially.
- Training IDs are unique and span 0–699,634. Test IDs are unique and span 699,635–999,478. The sets are disjoint. Exclude `id` from model inputs initially.
- Four predictors are categorical: `Gender`, `Customer Type`, `Type of Travel`, and `Class`. Their observed cardinalities are two, two, two, and three. No test-only categories were found.
- The other 17 predictors are numeric: age, flight distance, 13 service/convenience ratings, and departure/arrival delays. Ratings generally span 0–5; baggage handling spans 1–5.
- `Arrival Delay in Minutes` has 292 blank training cells (0.0417%) and 130 blank test cells (0.0434%). Other columns had no blanks or the common missing-value tokens checked. Blank means unavailable, not a recorded zero-minute delay. Pandas represents these blanks as NaN. The docs do not explain their cause.
- Feature-row hashing detected no duplicate feature rows within either split, no exact feature matches across train/test, and no conflicting-label duplicate groups. This is an exact-match screen, not proof that near-duplicate or generator-related leakage is absent.
- Inspected numeric means and low-cardinality frequencies are close across train/test. This supports an initial random split but does not establish identical distributions.
- Target prevalence by training ID decile is approximately stable. ID/target Pearson correlation is about 0.000125; this does not prove that every possible ID artifact is absent.
- `Online boarding` has the strongest measured numeric Pearson correlation with the target, about 0.560. This is association, not causal evidence or model feature importance.
- The combined loaded pandas frames for train/test used approximately 413 million bytes. Training can require substantially more memory. The inspection host had two CPU cores and about 2.1 GiB available memory; this is an observation, not a guaranteed future resource budget.

## Validation policy

1. Use a stratified 80/20 training/validation split with seed 42. All initial models must use the same rows.
2. Fit imputers, scalers, encoders, and any target-derived transforms only on the training partition. Do not use the outer validation partition for early stopping; use an internal split from the training partition.
3. Report ROC AUC as the selection metric. Report log loss and elapsed training time as secondary diagnostics; check convergence and peak memory where practical.
4. Start with untouched test data for inference only. Do not hand-label test rows or use leaderboard feedback as a substitute for local validation.
5. Recheck shortlisted improvements on another fixed seed or three-fold stratified validation before choosing a final model. If future checks reveal related-row groups, reassess the split before trusting scores.
6. Treat small single-split gains as uncertain. Prefer the simpler/faster model when extra complexity has no repeatable benefit. Do not promise an AUC before measurement.

## Model sequence

### 1. Logistic regression baseline

- One-hot encode categorical predictors with explicit handling of unseen categories.
- Impute missing arrival delay using the training-partition median; scale numeric predictors using training-partition statistics.
- Keep ratings numeric initially. Use a scikit-learn pipeline to prevent preprocessing leakage.
- Fit without class weights initially. Check convergence rather than silently accepting an unfinished fit.
- Include constant-probability predictions at training-partition prevalence as a sanity reference, not as a serious contender.

Purpose: establish an inexpensive, understandable reference and exercise the complete validation/prediction path.

### 2. Histogram gradient boosting

- Use scikit-learn `HistGradientBoostingClassifier` as the first nonlinear contender.
- Encode the four nominal features with training-fitted category mappings and mark their transformed columns explicitly as categorical. Handle unseen categories safely.
- Keep missing arrival delays as NaN. Do not scale numeric inputs unnecessarily.
- Start with bounded iterations/model size, CPU threads, and an internal early-stopping split. Compare on the unchanged outer holdout.

Purpose: capture interactions among loyalty, travel type, cabin class, and ratings without manual interaction expansion or another boosting dependency.

### 3. CatBoost challenger

The bounded challenger was run after the previous stages were measured and authorized. It used the project-local environment, native categorical handling, native numeric NaNs, bounded CPU threads, and an explicit training-only early-stopping split. The single fixed configuration qualified on both seeded holdouts. No unrestricted search was run.

CatBoost is plausible for this feature mix, not a guaranteed winner. LightGBM is an alternative later; do not add multiple boosting libraries at once.

## Conditional experiments and stopping rules

- Compare numeric versus categorical treatment of ratings; do not assume that zero means missing or not applicable.
- Try a few interpretable aggregates such as low-rating counts, mean service rating, and delay indicators. Retain them only after repeatable validation gains.
- Consider the public original dataset as a separate, compliant experiment after a synthetic-only baseline. Check schema, distribution, overlaps, and validation implications before combining datasets. Do not download it automatically.
- Blend only if aligned validation predictions improve ROC AUC consistently. Do not stack by default.
- Stop tuning when measured improvements do not justify added runtime/complexity or when the submission deadline requires finalization.

## Optional higher-compute track

Activate this track only after baseline and first histogram-boosting results exist, access to the proposed M4 Max Mac Studio with 64 GB unified memory is confirmed, and time remains before finalization. Access and speedups are not yet verified. The original VPS-only path remains complete without this track; skip these experiments rather than delay delivery.

- Reproduce the baseline on the Mac with an isolated native ARM64 environment, matching input fingerprints, split membership, and recorded dependency versions. Benchmark runtime and memory before setting concurrency. Limit concurrent fits multiplied by threads per fit to a measured CPU budget; leave memory headroom. Do not assume the VPS two-thread limit applies to the Mac.
- Upgrade shortlisted comparisons to five-fold stratified cross-validation using identical folds across models. Fit preprocessing and early stopping inside each fold's training partition only. Save out-of-fold probabilities indexed by row ID, per-fold AUC/log loss, overall out-of-fold scores, runtime, and peak memory. Repeat promising comparisons with another split seed if the budget permits; fold variation is a stability diagnostic, not a formal confidence interval.
- Add CatBoost and then LightGBM only when earlier results justify a challenger. Set a fixed trial-count and wall-clock budget before each search. Compare tree complexity, learning rate, and regularization within that budget; install dependencies incrementally. Stop configurations that exceed resource limits or show no useful improvement.
- Compute permutation importance on held-out data and inspect error slices by travel type, cabin class, loyalty, and age bands. Correlated predictors can share importance; slice differences are not causal findings. Keep rating encodings, small feature aggregates, and original-source data experiments under the existing conditional checks.
- Test a simple blend of complementary models using row-aligned out-of-fold predictions. Restrict weight exploration and confirm promising blends on another seed or untouched confirmation split; optimizing weights and reporting the same out-of-fold score alone is optimistic. Keep the single model if the gain is not repeatable.
- Optional learning-only extension: train a small tabular neural network with PyTorch's MPS backend after confirming backend availability and operation support. Fit encoding/scaling/imputation within each training partition. Benchmark against CPU and the tree baseline; unified memory is shared, not separate dedicated GPU RAM. Cap training time and stop if it adds neither useful learning nor complementary predictions. CatBoost GPU training uses CUDA, not MPS; do not assume Apple GPU acceleration for tree libraries. References: [CatBoost Apple GPU support request](https://github.com/catboost/catboost/issues/3095), [PyTorch MPS overview](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/). Verify installed-version support before execution.

Keep all transfers to an explicitly authorized machine under the same data-access restrictions. Do not put CSVs in Git to move them. Record measured results and skip reasons here; no additional proposal or architecture document is needed. This track does not authorize remote access, transfers, installations, or training by itself.

## Deliverable and scope

Use one local training entry point, one small runnable regression check, and one modest challenger entry point. Pin the environment and record seeds, settings, input fingerprints, scores, and runtime in lightweight local artifacts. Do not build an experiment platform.

After model selection, refit preprocessing/model on all training rows using settings/iteration counts selected without test labels. Generate probabilities in original test ID order. Check exact schema, row count, unique matching IDs, finite values in [0, 1], and absence of an accidental index column. Keep models, predictions, caches, and data out of Git unless separately authorized.

Skip GPU training and neural networks on the default VPS-only path; the optional higher-compute track permits a bounded MPS learning experiment. Skip AutoML sweeps, stacking infrastructure, dashboards, serving APIs, and a separate `design.md`. Add complexity only in response to measured need. This document and [plan.md](plan.md) do not authorize code implementation, training, committing, pushing, or submitting.
