# Modeling approach

## Status and objective

The VPS-only implementation and local training/export run completed on 2026-10-23. Bounded challenger experiments and the authorized fixed 50/50 CatBoost/LightGBM blend confirmation completed locally on 2026-10-24 HKT (2026-10-23 UTC). Real validation results and artifact paths are recorded below. No competition submission was uploaded.

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

## Bounded LightGBM experiment

On 2026-10-24 HKT, the user authorized one bounded LightGBM experiment after
the prior CatBoost work. The first completed run used the same one native
`lightgbm.LGBMClassifier` configuration, but its binary objective registered
both AUC and `binary_logloss` even though `eval_metric="auc"` was passed.
That run is not authoritative for the stopping-policy review. Its files were
preserved unchanged by renaming `artifacts/lightgbm/` to
`artifacts/lightgbm_dual_metric_review/`; they were not deleted or edited.

The corrected run below used an external 900-second hard timeout. It used one
native configuration only; there was no sweep, feature engineering, ensemble,
or submission. Numerical thread limits were set before importing NumPy,
pandas, or LightGBM. The environment installed only `lightgbm==4.6.0` into
`.venv` with `--no-cache-dir`, and the pin is in `requirements.txt`.

The corrected predeclared configuration was `objective=binary`, `metric=auc`,
`n_estimators=1500`, `learning_rate=0.04`, `num_leaves=31`, `max_depth=-1`,
`min_child_samples=40`, `reg_lambda=1`, `subsample=1`,
`colsample_bytree=1`, `random_state=42`, `n_jobs=2`, `deterministic=true`,
`force_col_wise=true`, and `verbosity=-1`. Each validation fit used AUC with
`early_stopping(100)` and `log_evaluation(0)` callbacks. A synthetic fit
verified that `evals_result_` contains only `auc`; the callback's
`first_metric_only=false` is therefore harmless because there is only one
registered metric. Category levels were fit from the inner training rows only;
unseen categories aligned to missing categorical values, categorical missing
values remained missing, and numeric NaNs remained numeric NaNs. `id` and
`satisfaction` were excluded.

The existing recorded outer fit/holdout arrays for seeds 42 and 2024 were
reused. Each LightGBM fit used the exact CatBoost depth-6 inner 90/10 split
from `artifacts/challengers/compare_metadata.json` for early stopping. The
outer holdout was never passed as `eval_set`.

| Split seed | Depth-6 CatBoost AUC | LightGBM AUC | Gain | Log loss | Best / actual iterations | Fit seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9573498661 | 0.9574892899 | +0.0001394238 | 0.2310565310 | 879 / 879 | 34.422 |
| 2024 | 0.9577455323 | 0.9576973220 | -0.0000482102 | 0.2298882161 | 783 / 783 | 28.837 |

The corrected smoke fit used 20,000 recorded inner-fit rows and 2,000
recorded inner-validation rows, with disjoint subsets. It completed in 0.570
seconds with 219 trees. Process high-water RSS was 972.9 MB after the first
fit and 1042.9 MB after the second; these are process-level measurements, not
isolated model allocations. CPU count and configured thread limit were both
two. Total elapsed time was 78.976 seconds. The authoritative metadata is
[artifacts/lightgbm/compare_metadata.json](../artifacts/lightgbm/compare_metadata.json), with validation predictions at
`artifacts/lightgbm/validation_lightgbm_seed42.csv` and
`artifacts/lightgbm/validation_lightgbm_seed2024.csv`.

The candidate was rejected because neither split met the predeclared minimum
gain of 0.0002 AUC. The decision was to retain the existing depth-6 CatBoost
candidate. No all-row LightGBM refit, native model, feature spec, or
`submission_lightgbm.csv` was created. The metadata records the completed
rejection, both validation results, input hashes before/after, source/version
and configuration hashes, the single-metric policy, fit resources, and
protected-artifact hashes. The fixed pre-existing artifact snapshot contained
24 files, including `deeper_catboost/` and the
`lightgbm_dual_metric_review/` archive; all before/after hashes matched.

The LightGBM run was validated with these final source-gate and run commands:

```text
.venv/bin/ruff format lightgbm_experiment.py test_train.py
.venv/bin/ruff check lightgbm_experiment.py test_train.py
.venv/bin/mypy lightgbm_experiment.py test_train.py
.venv/bin/python -m unittest -v test_train.py
/usr/bin/timeout --signal=TERM --kill-after=30s 900s .venv/bin/python lightgbm_experiment.py --data-dir data --artifacts-dir artifacts
```

The full test module ran 27 tests successfully. Input SHA-256 values remained
`train.csv` `2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36`
and `test.csv`
`ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6`; the
metadata reports protected baseline artifacts unchanged. The user-reported
public CatBoost result of AUC `0.95797`, rank `2225` remains separate from
these local validation scores and was not used for this decision. No agent
submission, commit, push, or publication occurred.

## Fixed 50/50 blend confirmation

On 2026-10-24 HKT, the user authorized one fresh-split confirmation of a fixed
`0.5` CatBoost / `0.5` LightGBM blend. This was a single predeclared blend
experiment: weights, model settings, and validation protocol were not searched.
The run used an external 1200-second timeout, two threads, and no new
dependencies. Its complete outputs are under
`artifacts/blend_50_50/`; existing artifacts were protected by the complete
LightGBM snapshot helper and remained unchanged.

The fresh outer split was stratified 80/20 with seed `31415`. Both models used
the same outer fit/holdout. Both used the same stratified 90/10 inner split
with seed `42`, drawn only from the fresh outer-fit rows; the outer holdout was
never used for fitting, category-level fitting, or early stopping. The split
arrays are stored in compressed
`artifacts/blend_50_50/fresh_split_31415.npz`, not as JSON row-ID lists.
CatBoost used the existing native depth-6 configuration: 1200 iterations,
learning rate `0.06`, native categorical columns, and early stopping 100 on
the inner evaluation rows. LightGBM used the existing explicit AUC-only
configuration: 1500 estimators, learning rate `0.04`, 31 leaves, and early
stopping 100 on the same inner evaluation rows. Numeric NaNs remained native;
LightGBM category levels were fitted only on its inner-fit rows.

| Candidate | Fresh ROC AUC | Fresh log loss | Actual / best iteration count | Fit seconds |
| --- | ---: | ---: | ---: | ---: |
| CatBoost depth 6 | 0.9584760730 | 0.2278616125 | 1199 / 1199 (best index 1198) | 179.258 |
| LightGBM | 0.9585916004 | 0.2275384364 | 778 / 778 (one-based) | 27.026 |
| Fixed 50/50 blend | 0.9588362995 | 0.2269662954 | weights only | — |

The fresh blend gained `+0.0003602265` AUC over fresh CatBoost and strictly
exceeded fresh LightGBM by `+0.0002446992`, so the predeclared confirmation
gate passed. The prior saved validation predictions were independently
reloaded and aligned by unique IDs before recomputing the two earlier blend
checks: seed 42 blend AUC `0.957713923393243` versus CatBoost
`0.957349866089771` (gain `+0.0003640573`), and seed 2024 blend AUC
`0.9580214315945633` versus CatBoost `0.9577455322589059` (gain
`+0.0002758993`). These earlier and fresh holdouts overlap, so the fresh
seed is additional stability evidence, not statistically independent
confirmation.

Because the gate passed, the existing all-train CatBoost model,
`artifacts/challengers/final_model_catboost.cbm`, feature spec, submission, and
provenance were reused unchanged. Its predictions matched the prior
submission within `2.22e-16`. LightGBM was refit on all training rows without
an eval set or early stopping using the median one-based best-iteration counts
`[879, 783, 778]`, giving `783` trees. The native model and fit-only category
spec are `artifacts/blend_50_50/final_model_lightgbm.txt` and
`artifacts/blend_50_50/feature_spec.json`. The final export is
`artifacts/blend_50_50/submission_catboost_lightgbm_50_50.csv`, with exactly
299,844 rows, original test ID order, `id,satisfaction` columns, and finite
probabilities in `[0, 1]`. No standalone LightGBM submission was created.

Clean-interpreter reload of both native models reconstructed all blend
predictions with maximum component difference `0.0` and maximum CSV blend
difference `2.22e-16`. Total experiment elapsed time was `260.930` seconds.
Input hashes remained train `2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36`
and test `ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6`.
The user-reported public CatBoost AUC `0.95797` / rank `2225` remains a
separate public result, not a local validation score and not evidence from
this run. No agent upload occurred. The user subsequently reported manually submitting the fixed blend and receiving public AUC `0.95817`, rank `2101`; this is user-reported, not independently fetched. It supersedes the reported depth-6 public best of `0.95797` and is not a local validation score.

After this completed run, the blend wrapper received guard-only hardening:
prior CatBoost/LightGBM provenance, fixed configurations, feature specs, and
iteration-count structure are checked before reuse; prior saved-prediction
blend gains are enforced; and input/protected-artifact drift is persisted as a
failure and raised without implying a valid selection or final output. Small
synthetic/mocked regressions cover valid metadata, changed hashes/configuration/
seeds/counts, and both integrity paths. The completed artifacts were not
retrained or modified, and no real experiment was rerun after these guard-only
changes. Final wrapper checks also cover equivalent relative/absolute artifact paths and repeated metadata-write failure without masking the original fit error. Formatting, lint, typecheck, and all 36 tests pass; model predictions and the submission file are unchanged.

The exact reproduction command was:

```text
/usr/bin/timeout --signal=TERM --kill-after=30s 1200s .venv/bin/python blend_experiment.py --data-dir data --artifacts-dir artifacts
```

## Fixed 75/25 blend experiment

On 2026-10-25 HKT, the authorized first phase scored one fixed candidate using
the saved slow-CatBoost validation probabilities at weight `0.75` and the
saved corrected original LightGBM probabilities at weight `0.25`. No model was
fit and no weights were searched. Both components were reloaded and aligned by
unique `id`; the current `0.5`/`0.5` blend was recomputed from the same arrays
before comparison.

| Split seed | Current 50/50 AUC | Fixed 75/25 AUC | Gain vs current | Decision |
| ---: | ---: | ---: | ---: | --- |
| 42 | 0.9577942088 | 0.9577166585 | -0.0000775503 | reject |
| 2024 | 0.9581967799 | 0.9581871345 | -0.0000096454 | reject |

The predeclared gate required at least `+0.0002` ROC AUC on both fixed
holdouts. It failed on both seeds, so the current 50/50 blend remains selected.
The overlapping holdouts are stability evidence, not independent confidence.
Validation outputs and complete pre-existing-artifact/input fingerprints are in
`artifacts/blend_75_25/`. The output directory contains only metadata, a
protected-artifact hash file, and the two row-aligned candidate validation CSVs.
No training, refit, submission export, Kaggle upload, or new dependency was
used.

The phase-1 gates passed in order:

```text
.venv/bin/ruff format weighted_blend_experiment.py test_train.py
.venv/bin/ruff check weighted_blend_experiment.py test_train.py
.venv/bin/mypy weighted_blend_experiment.py test_train.py
.venv/bin/python -m unittest -v test_train.py
.venv/bin/python weighted_blend_experiment.py --data-dir data --artifacts-dir artifacts
```

The phase-1 source and documentation change was committed and pushed before
the separate memory-bounded RandomForest phase. The exact commit is recorded
in the repository history and was verified against `origin/main`.

## Fixed memory-bounded RandomForest experiment

On 2026-10-25 HKT, after the phase-1 commit and push, the authorized RF phase
ran exactly one fixed configuration sequentially on the recorded seed-42 and
seed-2024 holdouts. The configuration was
`n_estimators=200`, `max_depth=14`, `max_leaf_nodes=1024`,
`min_samples_leaf=30`, `max_features="sqrt"`, `bootstrap=true`,
`max_samples=0.7`, `random_state=42`, `n_jobs=2`, and `class_weight=None`.
The structural node bound was `200 * (2 * 1024 - 1) = 409,400` nodes.

Preprocessing was fit only on the exact 503,737-row competition inner-fit
partition for each seed. It used median numeric imputation, most-frequent
categorical imputation, one-hot encoding with `handle_unknown="ignore"`, and a
dense float32 output with 26 transformed features. The 55,971-row inner
validation partition was excluded from RF fitting because RF had no early
stopping; the 139,927-row outer holdout was used only for evaluation. The
current blend was recomputed from the saved slow-CatBoost/original-LightGBM
predictions by unique ID, then combined as `0.90 * current_50_50 + 0.10 * RF`.

| Split seed | Standalone RF AUC / log loss | Current 50/50 AUC | Fixed 90/10 RF blend AUC / log loss | Gain vs current | Fit seconds | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9550480356 / 0.2411610720 | 0.9577942088 | 0.9577360938 / 0.2303499831 | -0.0000581150 | 38.808 | 961.9 MB |
| 2024 | 0.9554150226 / 0.2400712685 | 0.9581967799 | 0.9581036501 / 0.2291926925 | -0.0000931299 | 39.200 | 1,019.3 MB |

The predeclared gate required at least `+0.0002` AUC on both seeds. It failed
on both seeds, so the current 50/50 blend remains selected. Each fitted forest
used 200 trees and reached the recorded 2,047-node per-tree maximum; the total
observed node count was 409,400, equal to the structural bound. The final
experiment took 89.172 seconds, used two numerical threads, and wrote no model
files or test prediction export.

The authoritative outputs are under `artifacts/random_forest/` and contain two
row-aligned validation CSVs plus metadata and protected-artifact hashes totaling
20,922,561 bytes. The initial implementation attempt is preserved separately
under `artifacts/random_forest_initial_failure/`; it fit seed 42 but failed
after fitting while recording a scikit-learn-incompatible `max_features_`
attribute. That failure was corrected and the fixed configuration was rerun;
it was not an OOM or timeout retry. The successful run recorded all four data
CSV hashes unchanged and all pre-existing artifact hashes unchanged.

The final RF source gates passed in order:

```text
.venv/bin/ruff format random_forest_experiment.py test_train.py
.venv/bin/ruff check random_forest_experiment.py test_train.py
.venv/bin/mypy random_forest_experiment.py test_train.py
.venv/bin/python -m unittest -v test_train.py
/usr/bin/timeout --signal=TERM --kill-after=30s 10800s .venv/bin/python random_forest_experiment.py --data-dir data --artifacts-dir artifacts
```

## Fixed five-feature pass

On 2026-10-24 HKT, the authorized feature pass tested exactly five appended
numeric predictors: `rating_mean`, `low_rating_count`, `departure_delayed`,
`arrival_delayed`, and `arrival_delay_missing`. The mean and low-rating count
use the explicit 13 service-rating fields. Observed zero ratings count as low
ratings. Delay indicators preserve missing delay values as missing; the
missingness indicator is separate. The four nominal predictors, native
categorical handling, numeric NaNs, CatBoost depth-6 settings, and explicit
AUC-only LightGBM settings were unchanged. Input and every pre-existing
artifact fingerprint were checked before and after the run.

The run used the recorded outer seed-42 and seed-2024 arrays and the prior
CatBoost 90/10 seed-42 inner arrays. Fits were sequential with two threads.
The current unengineered blend was recomputed from the saved CatBoost and
LightGBM validation predictions after unique-ID alignment. The predeclared
gate required an engineered fixed 50/50 blend gain of at least `0.0002` on
both seeds.

| Split seed | Feature CatBoost AUC | Feature LightGBM AUC | Feature blend AUC | Gain vs current blend | Feature blend log loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9573208451 | 0.9573044042 | 0.9575926818 | -0.0001212416 | 0.2304502788 |
| 2024 | 0.9576483851 | 0.9577904776 | 0.9580344291 | +0.0000129975 | 0.2292844251 |

The current unengineered blend AUCs were `0.957713923393243` and
`0.9580214315945633`. Neither engineered gain met `0.0002`, so the pass was
rejected after the two recorded seeds. The seed-31415 confirmation, all-row
refits, native model reload, and descriptive final submission were skipped.
No loser model or submission was exported. New validation and provenance
artifacts are under `artifacts/feature_pass/`; all existing data and artifact
hashes remained unchanged. The run took 499.038 seconds and ended with
`selection.decision=retain_unengineered_blend`.

The real experiment used the model code and feature bundle described above. Subsequent wrapper-only fixes persist source/dependency discovery failures, resolve reload paths before changing subprocess working directory, and clear the completed fresh-fit status before its gate. The model logic, input files, validation outputs, and selected current blend are unchanged; no training rerun was needed for these guard-only fixes. Formatting, lint, typecheck, and all 42 tests pass on the final source. The executed training command and current gate commands are:

```text
.venv/bin/ruff format train.py test_train.py compare.py lightgbm_experiment.py blend_experiment.py feature_experiment.py
.venv/bin/ruff check train.py test_train.py compare.py lightgbm_experiment.py blend_experiment.py feature_experiment.py
.venv/bin/mypy train.py test_train.py compare.py lightgbm_experiment.py blend_experiment.py feature_experiment.py
.venv/bin/python -m unittest -v test_train.py
/usr/bin/timeout --signal=TERM --kill-after=30s 1500s .venv/bin/python feature_experiment.py --data-dir data --artifacts-dir artifacts
```

## Longer CatBoost experiment

On 2026-10-24 HKT, the user authorized one longer-CatBoost experiment. The new
entry point was `longer_catboost.py`, run with a 2400-second external timeout
and an internal 40-minute deadline. It changed only the CatBoost iteration cap
from `1200` to `3000`; depth `6`, learning rate `0.06`, `l2_leaf_reg=3`,
`Logloss`/`AUC`, seed `42`, patience `100`, two threads, and
`allow_writing_files=false` were unchanged. Fits were sequential. Existing
recorded outer arrays, prior CatBoost inner arrays, corrected LightGBM
validation probabilities, and the fixed 50/50 weights were reused. LightGBM
was not refit.

| Split seed | Longer CatBoost AUC | Fixed 50/50 blend AUC | Gain vs current blend | Actual / best iteration |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9573498661 | 0.9577139234 | +0.0000000000 | 1171 / 1170 |
| 2024 | 0.9579485233 | 0.9581305578 | +0.0001091262 | 1719 / 1718 |

The predeclared advancement gate required a new blend gain of at least
`0.0002` on both seeds. It failed on both seeds, so the current blend was
retained. Fresh seed `31415`, the conditional all-training-row CatBoost refit,
native model reload, and descriptive final submission were skipped. No loser
submission was written. The validation artifacts with row IDs and CatBoost,
reused LightGBM, and blend probabilities are under
`artifacts/longer_catboost/`; the pre-guard metadata recorded `20,892,065`
bytes, below the `200 MiB` budget.

Input hashes matched before and after, every pre-existing artifact hash matched,
and `current_fit`, `fresh_validation`, and `final` are all clear in the
completed metadata. The run recorded CatBoost fit times of `200.527` seconds
and `275.663` seconds, with total elapsed time `484.547` seconds. The local
source gates and regression suite passed with 45 tests before the guard-only
review fixes:

```text
.venv/bin/ruff format longer_catboost.py test_train.py
.venv/bin/ruff check longer_catboost.py test_train.py
.venv/bin/mypy longer_catboost.py test_train.py
.venv/bin/python -m unittest -v test_train.py
/usr/bin/timeout --signal=TERM --kill-after=30s 2400s .venv/bin/python longer_catboost.py --data-dir data --artifacts-dir artifacts
```

An independent review then identified wrapper-only issues in saved-prediction
alignment, project-relative artifact paths, timeout cleanup, iteration-count
reuse, and post-run size accounting. These fixes do not change model settings,
selected models, validation outputs, or the completed result. The real run was
not rerun; its artifacts remain byte-identical, and their `source_sha256`
records remain the historical pre-guard source hashes. The final guard-only
regression suite passes with 50 tests.

## Slower-learning experiment

On 2026-10-24 HKT the user authorized one slower-learning pass after the
longer-iteration cap produced no gain. The new entry point was
`slower_learning.py`. It changed learning rates and caps only: LightGBM
`0.04 → 0.02` with `1500 → 3000` trees, CatBoost `0.06 → 0.03` with
`1200 → 6000` iterations, and patience `100 → 200` for each. Depth, leaf
count, `l2_leaf_reg`, `Logloss`/`AUC`, seeds, and two threads were unchanged.
Both models were fitted on the recorded outer splits and the existing CatBoost
inner 90/10 split; the fixed 50/50 weights were reused. Seeds `42` and `2024`
were used. Three fixed blends were scored against the current blend:

| Blend | Seed 42 gain | Seed 2024 gain |
| --- | ---: | ---: |
| original CatBoost + slow LightGBM | -0.0000732 | +0.0000341 |
| slow CatBoost + original LightGBM | +0.0000803 | +0.0001753 |
| slow CatBoost + slow LightGBM | +0.0000229 | +0.0001822 |

All three failed the predeclared `0.0002` gain on both seeds, so the current
blend was retained and no fresh-split confirmation was run. Standalone scores
show why the components differed: slow CatBoost improved over the original on
both seeds (`0.9575292` vs `0.9573499`, and `0.9580269` vs `0.9577455`) using
`4498` and `4982` trees of `6000`, while slow LightGBM was mixed
(`0.9573779` vs `0.9574893`, and `0.9577325` vs `0.9576973`).

A paired bootstrap over the held-out rows estimated the best blend gain at
`+0.00008` with standard error `0.000039` on seed 42 and `+0.000175` with
standard error `0.000041` on seed 2024, with the gain positive in 98% and 100%
of resamples. The improvement is real but smaller than the gate, which was
predeclared to prevent chasing unpaired noise.

The user then explicitly authorized exporting that below-gate candidate. The
slow CatBoost was refit on all training rows at the rounded median one-based
count `4740` from `4498` and `4982` with early stopping disabled; the existing
783-tree LightGBM model was reused unchanged. The export is
`artifacts/slower_learning/submission_slowcat_lightgbm_50_50.csv` with 299,844
rows in exact test ID order, finite values from `0.0020123` to `0.9929639`.
Fresh-process reloads matched the in-process component predictions exactly and
the CSV round-trip difference was `2.22e-16`. The directory holds 32,213,544
bytes; 43 prior artifact hashes and both input hashes were unchanged. The
guard-only regression suite passes 52 tests. A first export attempt failed its
LightGBM reload because the raw test frame including `id` was passed to the
prediction helper; the model was already fitted and saved, so the refit was
recovered from that file instead of being repeated, and its fit time is
recorded as not measured.

## Native categorical service-rating experiment

On 2026-10-24/25 HKT, the authorized categorical-ratings experiment used the
existing slow-CatBoost settings: depth `6`, learning rate `0.03`, `6000`
iterations, patience `200`, `Logloss`/`AUC`, model seed `42`, two threads, and
`allow_writing_files=false`. It changed only the 13 service-rating columns from
numeric to native categorical features. The four existing nominal columns
remained categorical, for 17 native categorical columns total. Ratings were
copied without mutation, canonicalized to stable integer tokens, treated
observed zero as valid, and rejected when missing, non-integral, non-numeric, or
outside `0..5`. Numeric delay NaNs were preserved; nominal missing values use
`<missing>`.

The earlier two-seed attempt remains unchanged in
`artifacts/categorical_ratings/`. It completed seed `42` but was externally
interrupted during seed `2024`; its metadata honestly retains status `failed`
and the original recorded error `KeyboardInterrupt` with an empty message. Its
valid seed-42 CSV was reused only after verifying its SHA-256, source/config
provenance, input hashes, recorded arrays, IDs, and probabilities. Seed `42`
was not retrained.

| Split seed | Categorical CatBoost AUC | Candidate 50/50 blend AUC | Gain vs current blend | Trees / best count | Fit seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9575978195 | 0.9578470100 | +0.0000528012 | 4142 / 4142 | 2264.340 |
| 2024 | 0.9580262339 | 0.9581904783 | -0.0000063016 | 5383 / 5383 | 2927.894 |

The second-split run used the explicit `--complete-second-split` mode with an
internal `5100`-second (85-minute) alarm and the externally authorized command
bound of `5400` seconds (90 minutes), without changing parameters or launching
another seed. Its validation file is
`artifacts/categorical_ratings_seed2024/validation_categorical_ratings_seed2024.csv`.
The new metadata is
`artifacts/categorical_ratings_seed2024/compare_metadata.json`; it combines the
new result with a lineage reference to the verified prior seed-42 result. The
seed-2024 fit used 5383 trees, took `2927.894` fit seconds and `2934.498`
elapsed seconds, and observed a `1898.1 MB` maximum RSS. The new output is
`10,467,631` bytes, below the `100 MiB` budget.

The original completed validation file remains
`artifacts/categorical_ratings/validation_categorical_ratings_seed42.csv`.
The original failed-run metadata remains
`artifacts/categorical_ratings/compare_metadata.json`; it records the exact 17
categorical columns, settings, source/dependency hashes, split-array
fingerprints, fit resource observations, and its failed state. It contains
`10,452,622` bytes across one validation CSV and metadata, below the `100 MiB`
budget. All 50 pre-existing artifact hashes, both input hashes, and source
hashes matched before and after the second-split run. The combined gate still
requires `+0.0002` on each seed and fails on both seeds, so this run is
information-only; the current blend remains selected. No refit, export,
submission, or upload was attempted.

The source gates and full regression suite passed before execution:

```text
.venv/bin/ruff format blend_experiment.py categorical_ratings.py compare.py feature_experiment.py lightgbm_experiment.py longer_catboost.py slower_learning.py train.py test_train.py
.venv/bin/ruff check blend_experiment.py categorical_ratings.py compare.py feature_experiment.py lightgbm_experiment.py longer_catboost.py slower_learning.py train.py test_train.py
.venv/bin/mypy blend_experiment.py categorical_ratings.py compare.py feature_experiment.py lightgbm_experiment.py longer_catboost.py slower_learning.py train.py test_train.py
.venv/bin/python -m unittest -v test_train.py
/usr/bin/timeout --signal=TERM --kill-after=30s 5400s .venv/bin/python categorical_ratings.py --complete-second-split --data-dir data --artifacts-dir artifacts
```

## Original-source augmentation experiment

On 2026-10-25 HKT, the user authorized one bounded source-data augmentation
experiment. The source was the downloaded Kaggle dataset
`arseniyshutko/binary-aviation-satisfaction-129k`; its saved metadata states
CC0-1.0 and identifies educational AI/ML course material as the provenance.
The source is public external data, but its relationship to the competition
generator is uncertain and its marginal distributions differ materially.
The read-only assessment recorded 129,880 rows, 17 exact feature overlaps with
train, 4 with test, and 3 train-label disagreements among those overlaps. The
run excluded overlaps using only all-21-feature canonical equality, never the
IDs or labels; it independently found 129,859 clean source rows, zero source
feature duplicates, zero conflicting source labels, and removed no duplicate
rows.

The entry point was `source_augmentation.py`. It reused the recorded outer
splits and the prior CatBoost seed-42 90/10 inner arrays after recomputing and
verifying them with the existing helper. It used only the existing four native
nominal columns; all 13 ratings stayed numeric, observed zeros stayed valid,
and numeric arrival-delay NaNs remained NaN. The exact slow-CatBoost settings
were unchanged: depth `6`, `6000` iterations, learning rate `0.03`,
`l2_leaf_reg=3`, `Logloss`/`AUC`, seed `42`, patience `200`, two threads, and
`allow_writing_files=false`. Each fit used 503,737 competition inner-fit rows
plus all 129,859 clean source rows at unit weight. Early stopping used only
the 55,971 competition inner-validation rows; the 139,927 outer holdout rows
were used only for scoring. No source rows entered early stopping or scoring,
and no origin indicator, resampling, weight search, refit, export, or third
seed was run.

The fixed 50/50 candidate used augmented CatBoost predictions with the saved
corrected original LightGBM predictions. Baselines were recomputed after
unique-ID alignment and finite-probability checks before each expensive fit.

| Split seed | Augmented CatBoost AUC / log loss | Candidate blend AUC / log loss | Gain vs current blend | Trees | Fit seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9573326966 / 0.2311362498 | 0.9577846015 / 0.2298503421 | -0.0000096073 | 4444 | 864.346 |
| 2024 | 0.9573831197 / 0.2305897936 | 0.9579044878 / 0.2291248495 | -0.0002922921 | 2073 | 415.016 |

The current slow-CatBoost plus original-LightGBM baselines were
`0.9577942088` and `0.9581967799`. The predeclared `+0.0002` AUC gate failed
on both overlapping fixed holdouts, so the decision is
`retain_current_blend`. The overlapping holdouts are stability evidence, not
independent confirmation. No automatic third seed, refit, export, Kaggle
upload, or commit/push was performed.

The external command was bounded to 10,800 seconds with a 30-second kill
grace; the entry point had an internal 10,500-second timeout. Both sequential
fits completed in 1,288.643 total seconds. Peak process RSS was 1,641.4 MB on
seed 42 and 1,699.1 MB on seed 2024; configured numerical and CatBoost thread
limits were two. The isolated output directory contains 20,904,032 bytes and
the two row-aligned component/LightGBM/blend validation files plus metadata.
All 54 pre-existing artifact-file hashes matched before and after. All four
data CSV hashes also matched before and after:

```text
sample_submission.csv  6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71
source_airline/data.csv 2ad76465f6115f93d16121891a6b1235acb8007a4ab065cae0dc46b60ad116e3
test.csv               ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6
train.csv              2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36
```

The authoritative metadata and validation outputs are under
`artifacts/source_augmentation/`. The source was not copied into artifacts,
uploaded, or sent to an external model. The pre-run gates passed in order:

```text
.venv/bin/ruff format source_augmentation.py test_train.py
.venv/bin/ruff check source_augmentation.py test_train.py
.venv/bin/mypy source_augmentation.py test_train.py
.venv/bin/python -m unittest -v test_train.py
/usr/bin/timeout --signal=TERM --kill-after=30s 10800s .venv/bin/python source_augmentation.py --data-dir data --artifacts-dir artifacts
```

## Fixed CatBoost seed-average experiment

On 2026-10-25 HKT, the user authorized one fixed two-seed averaging experiment.
The hypothesis was limited to training randomness: the existing slow CatBoost
configuration was built by `slower_learning.build_slow_catboost()` and then had
only `random_seed` changed from `42` to `2025`. Depth `6`, learning rate `0.03`,
`6000` iterations, patience `200`, `l2_leaf_reg=3`, `Logloss`/`AUC`, two
threads, native handling for the four nominal columns, numeric ratings, and
numeric NaN preservation were unchanged. The split seed remained `42` for the
existing 90/10 CatBoost inner partition.

The runner reused the exact saved outer arrays for seeds `42` and `2024`,
validated the prior CatBoost inner arrays through `make_catboost_internal_split`,
and fit only the `503,737` competition inner-fit rows. The `55,971` competition
inner-validation rows were used for early stopping; the `139,927` outer
holdout rows were used only for metrics. It reused the saved seed-42 slow
CatBoost and corrected LightGBM probabilities after exact row-index, ID, and
finite-probability checks. The fixed candidate was
`.25 CatBoost42 + .25 CatBoost2025 + .5 LightGBM`.

| Split seed | Seed-2025 CatBoost AUC / log loss | Averaged CatBoost AUC / log loss | Candidate blend AUC / log loss | Gain vs current 50/50 | Best count / actual trees | Fit seconds | Peak RSS MB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.9576054158 / 0.2306286322 | 0.9576015135 / 0.2305446511 | 0.9578276160 / 0.2300548582 | +0.0000334072 | 4365 / 4365 | 673.851 | 1550.6 |
| 2024 | 0.9581443754 / 0.2294766879 | 0.9581402066 / 0.2294053983 | 0.9582381409 / 0.2288764589 | +0.0000413610 | 5134 / 5134 | 790.482 | 1601.2 |

The current 50/50 baseline preflight recomputed as `0.9577942088132286` on
seed `42` and `0.9581967799255812` on seed `2024`. Both candidate gains were
positive but below the predeclared `+0.0002` requirement, so the decision was
`retain_current_blend`. The two holdouts overlap and are stability evidence,
not independent confirmation. No tuning after seeing results, third split,
refit, export, submission, Kaggle upload, or commit/push was performed.

The bounded command was:

```text
/usr/bin/timeout --signal=TERM --kill-after=30s 10800s .venv/bin/python catboost_seed_average.py --data-dir data --artifacts-dir artifacts
```

Both fits completed sequentially in `1,472.386` seconds. The isolated output
directory is `artifacts/catboost_seed_average/` and contains `47,469,400`
bytes, below the `100 MiB` budget. It contains only the new CatBoost
predictions, copied old CatBoost/LightGBM components, averaged CatBoost
predictions, fixed candidate predictions, and metadata; no native validation
model or forest artifact was created. Progress was recorded every 500 trees.

The runner recorded source/dependency hashes, split and ID/label fingerprints,
one-based `get_best_iteration()+1`, actual tree counts, fit times, resources,
and partial-fit state. All `67` pre-existing artifact-file hashes and all four
input CSV hashes were unchanged. The external source CSV was fingerprinted for
integrity only and was not used for training or features. The pre-run gates
passed in order and all `71` regression tests passed:

```text
.venv/bin/ruff format blend_experiment.py categorical_ratings.py catboost_seed_average.py compare.py feature_experiment.py lightgbm_experiment.py longer_catboost.py random_forest_experiment.py slower_learning.py source_augmentation.py train.py weighted_blend_experiment.py test_train.py
.venv/bin/ruff check blend_experiment.py categorical_ratings.py catboost_seed_average.py compare.py feature_experiment.py lightgbm_experiment.py longer_catboost.py random_forest_experiment.py slower_learning.py source_augmentation.py train.py weighted_blend_experiment.py test_train.py
.venv/bin/mypy blend_experiment.py categorical_ratings.py catboost_seed_average.py compare.py feature_experiment.py lightgbm_experiment.py longer_catboost.py random_forest_experiment.py slower_learning.py source_augmentation.py train.py weighted_blend_experiment.py test_train.py
.venv/bin/python -m unittest -v test_train.py
/usr/bin/timeout --signal=TERM --kill-after=30s 10800s .venv/bin/python catboost_seed_average.py --data-dir data --artifacts-dir artifacts
```

## Fixed five-fold cross-validation

On 2026-10-25 HKT the user authorized stronger validation for the one frozen
question: current `.5 CatBoost42 + .5 LightGBM` versus the
`.25 CatBoost42 + .25 CatBoost2025 + .5 LightGBM` candidate. Both strategies
were fixed before training and no alternative weights, seeds, or models were
added later.

`StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)` over all
`699,635` competition training rows produced five folds of `139,927` rows each.
Within every fold, the existing CatBoost inner 90/10 split with seed `42`
supplied `503,737` inner-fit rows and `55,971` early-stopping rows. Outer fold
holdout labels never entered preprocessing, fitting, or early stopping. Each
fold fitted three components: slow CatBoost with model seed `42`, the same
configuration with only `random_seed=2025`, and the original corrected
LightGBM. Fifteen sequential fits used two threads; fitted models were deleted
after their predictions were saved.

| Fold | Baseline AUC | Candidate AUC | Paired gain |
| ---: | ---: | ---: | ---: |
| 1 | 0.958922500 | 0.958943050 | +0.0000205 |
| 2 | 0.958767537 | 0.958779610 | +0.0000121 |
| 3 | 0.959275379 | 0.959265292 | -0.0000101 |
| 4 | 0.958566904 | 0.958580429 | +0.0000135 |
| 5 | 0.958799291 | 0.958808713 | +0.0000094 |

Mean fold AUC was `0.9588663` for the baseline and `0.9588754` for the
candidate. The mean paired gain was `+0.0000091` with descriptive standard
deviation `0.0000103`, four of five folds positive, and a pooled out-of-fold
gain of `+0.0000081` (`0.9588604` to `0.9588685`). The predeclared rule
required a mean gain of at least `0.0002`, at least four positive folds, and a
positive pooled gain. Only the direction conditions held, so the recorded
decision is `retain_current_blend`.

This is stronger stability evidence than repeated use of the same two seeded
holdouts, but the folds are not an untouched independent test: the same data
informed earlier choices, and pooled out-of-fold scores mix fold-specific
fits. No export, refit, submission, or Kaggle upload was performed.

The delegated worker hit a provider usage limit after 13 of 15 fits. The two
remaining fold-5 component fits were completed by reusing
`five_fold_cv.py` helpers with identical settings, and all five fold results
were recomputed from the saved component predictions before the pooled scores
were written. This completion path is recorded in the run metadata.

Completed out-of-fold predictions cover all `699,635` rows exactly once with
unique IDs in training order. The output directory
`artifacts/five_fold_cv/` holds `51,015,226` bytes, inside the `150 MiB`
budget. All `79` pre-existing artifact hashes and all four input CSV hashes
were unchanged. Formatting, Ruff, mypy, and all `75` regression tests pass.

## Fixed Eco/business/loyal specialist experiment

On 2026-10-26 HKT, the user authorized one fixed specialist experiment after
the existing five-fold OOF exploration. The frozen group was exactly
`Class=Eco`, `Type of Travel=Business travel`, and
`Customer Type=Loyal Customer` (`65,542` rows; `27,051` positive and `38,491`
negative labels). The experiment used the existing
`StratifiedKFold(n_splits=5, shuffle=true, random_state=2026)` membership and
the existing seed-42 90/10 inner split within each outer-fit partition. Each
specialist fit used only group rows from the inner fit; early stopping used
only group rows from the inner validation; predictions used only the group
rows in that fold's outer holdout. The outer holdout was never used for
training or early stopping, and the original external source CSV was not used.

The only model was `slower_learning.build_slow_catboost()` with seed `42`,
learning rate `0.03`, cap `6000`, depth `6`, `l2_leaf_reg=3`,
`Logloss`/`AUC`, patience `200`, native four nominal columns, numeric ratings
with valid zeros, numeric delay NaNs preserved, two threads, and
`allow_writing_files=false`. The candidate copied the frozen baseline outside
the group and used exactly `0.75 * baseline + 0.25 * specialist` inside it.

| Fold | Global baseline AUC | Global candidate AUC | Paired gain | Baseline log loss | Candidate log loss | Specialist fit seconds | Trees |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.958922500 | 0.958963938 | +0.000041438 | 0.226765545 | 0.226548909 | 6.947 | 791 |
| 2 | 0.958767537 | 0.958798956 | +0.000031419 | 0.227905161 | 0.227763440 | 7.932 | 954 |
| 3 | 0.959275379 | 0.959296485 | +0.000021106 | 0.226650963 | 0.226477808 | 6.887 | 764 |
| 4 | 0.958566904 | 0.958595984 | +0.000029080 | 0.227620340 | 0.227472713 | 9.330 | 1182 |
| 5 | 0.958799291 | 0.958846761 | +0.000047470 | 0.227168224 | 0.226977895 | 8.097 | 1038 |

Mean fold global AUC changed from `0.9588663222` to `0.9589004248`, a paired
gain of `+0.0000341026`; all five folds were positive. Pooled global OOF AUC
changed from `0.9588603718` to `0.9588946957` (`+0.0000343239`). Pooled global
log loss changed from `0.2272220463` to `0.2270481533` (`-0.0001738930`, lower
is better). The advancement rule required mean paired AUC gain at least
`0.0002`, at least four positive folds, and positive pooled gain. The first
two direction checks passed, but the minimum gain did not; the decision is
`retain_current_blend`. No ad-hoc probability correction, weight search, or
new model configuration was run.

For the specialist group, pooled AUC was `0.9328303257` baseline,
`0.9337368632` specialist, and `0.9335750473` for the routed candidate. Pooled
log loss was `0.3105689108`, `0.3077579580`, and `0.3087126712`, respectively.
Mean predicted probability versus actual prevalence was `0.4112803771` vs
`0.4127277166` for the baseline and `0.4116546543` vs `0.4127277166` for the
candidate. At the stated probability threshold `0.5`, subgroup error rates
were `0.1220744` baseline and `0.1212352` candidate. These are descriptive
calibration/error diagnostics, not label-noise or causal claims; global fold
metrics include cross-group ordering effects.

The five sequential fits completed in `48.302` seconds with two numerical
threads and process maximum RSS `656.3 MB`. Progress was saved every 500 trees
and models were deleted after prediction. The output is
`artifacts/eco_business_specialist/` (`14,454,347` bytes, below the `100 MiB`
budget) with five specialist component files, one compressed full candidate
OOF file, metadata, and no saved validation models. All `65,542` group rows
were assigned once; non-group candidate probabilities are bitwise unchanged.
All `107` pre-existing artifact hashes, all four input CSV hashes, source
hashes, feature/configuration hashes, and fold-membership hashes matched
before and after. The source gates ran in order and all `79` regression tests
passed. A pre-fit metadata-only runner failure was preserved separately under
`artifacts/eco_business_specialist_initial_failure/`; it trained no model.
No refit, export, submission, upload, commit, or push was performed.

## Authorized below-gate specialist export

After the specialist gate failed, the user explicitly authorized one local
descriptive export. The export entry point is
`eco_business_specialist_export.py`; it validates the completed five-fold
specialist metadata, exact group/configuration/source/input fingerprints, and
the saved row-aligned OOF provenance before fitting. It does not modify the
existing CV artifacts, the slow-learning baseline, or any submitted CSV.

The specialist was refit only on the exact training group
`Class=Eco`, `Type of Travel=Business travel`, and
`Customer Type=Loyal Customer`: 65,542 training rows. The five recorded
one-based CV counts were `[791, 954, 764, 1182, 1038]`; their metadata-derived
median selected `954` trees. The full fit used native CatBoost categorical
handling for the original four nominal columns, numeric NaNs unchanged, and
`learning_rate=0.03`, `depth=6`, `l2_leaf_reg=3`, `loss_function=Logloss`,
`eval_metric=AUC`, `random_seed=42`, two threads, and
`allow_writing_files=false`. Early stopping and an evaluation set were both
disabled for this full fit. The recorded fit time was 8.308 seconds and total
export runtime was 15.254 seconds.

The frozen baseline was
`artifacts/slower_learning/submission_slowcat_lightgbm_50_50.csv` with the
metadata-confirmed SHA-256
`5a5f985fd98a78bce5cf93e0a2dde6f228b36d25cc4d60c54b859be6d7262059`. The
candidate preserves that baseline probability text outside the test group and
uses `0.75 * baseline + 0.25 * specialist` only for the 28,227 matching test
rows. The exported CSV contains exactly 299,844 rows in test order, unique IDs,
finite probabilities in `[0, 1]`, and a maximum CSV numeric difference of
`0.0` from the in-process candidate. A clean subprocess reloaded the saved
native model/specification and independently reproduced all candidate
probabilities with maximum absolute difference `0.0`.

The export is explicitly designated
`below_gate_user_authorized_not_winner`; the recorded CV gate remains false
with decision `retain_current_blend`. The new artifacts are under
`artifacts/eco_business_specialist_export/`. The model SHA-256 is
`8f3cbde6329a1417ee5c7ce74b1bb2bea2572bd185e7d189c78ce3eccad452c1`, and the
descriptive CSV SHA-256 is
`5286a08fac14af0061dbb97f336018eec5de4629946f6120944f2c20e5eb990f`.
The directory size is 9,309,471 bytes, below the 100 MiB limit. All 115 prior
artifact-file hashes, all four CSV input hashes, and the source fingerprints
matched before and after. No public-score claim, Kaggle upload, commit, or push
was made.

## Fixed LightGBM capacity experiment

Tested only `num_leaves=31 -> 63`; all other original corrected-LightGBM
settings and fit-only category preparation stayed unchanged. The two original
outer holdouts and inner seed-42 partitions were reused. The candidate was
`0.5 * existing slow CatBoost + 0.5 * LightGBM63`, compared with the submitted
model configuration's slow-CatBoost/original-LightGBM blend.

- Seed 42: baseline AUC `0.9577942088132286`, candidate
  `0.957878606035489`, gain `+0.00008439722226039059`; LightGBM63
  standalone AUC `0.9576453875612927`, best/actual iterations `375`,
  fit time `18.922` seconds.
- Seed 2024: baseline AUC `0.9581967799255812`, candidate
  `0.958281830839665`, gain `+0.00008505091408383958`; LightGBM63
  standalone AUC `0.9578302011802371`, best/actual iterations `517`,
  fit time `22.414` seconds.
- Both gains miss the predeclared `+0.0002`-each-seed gate. Retain the current
  blend. These reused holdouts measure development stability, not independent
  confirmation or expected public-score gains.
- The real run took `52.855` seconds under internal/external limits of
  `1680/1800` seconds. Two threads; process peak RSS reached `897.0` MB.
  Output directory: `26,602,279` bytes. Independent recomputation verified
  prediction IDs, blend arithmetic, AUC/log loss, source hashes, all four input
  hashes and all 120 prior-artifact hashes. Format, Ruff, mypy and 85 tests pass.
- Reproduce: `timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python
  lightgbm_capacity.py --data-dir data --artifacts-dir artifacts` (one shell
  command). Outputs: `artifacts/lightgbm_capacity63/`.
- No new dependencies, CatBoost retraining, additional folds, final refit,
  model persistence, submission export, upload, commit or push occurred.
  Existing uncommitted specialist work was preserved.

## Fixed CPU XGBoost experiment

Installed and pinned `xgboost-cpu==3.2.0` locally; its build reports
`USE_CUDA=False`. No CUDA dependency was installed. Tested one fixed
`XGBClassifier`: CPU histogram trees, depth 6, learning rate `0.04`, cap
1500, minimum child weight 5, L2 1, L1 0, full row/column sampling,
256 histogram bins, depthwise growth, model seed 42, two threads, and
AUC-only early stopping with patience 100. The original 21 predictors and
four nominal categories were retained; numeric NaNs/zeros were preserved.
Category levels came only from the original inner-fit rows.

The original outer seeds 42/2024 and inner seed 42 were reused. The fixed
candidate was `0.8 * current blend + 0.2 * XGBoost`, equivalently
`0.4 * slow CatBoost + 0.4 * original LightGBM31 + 0.2 * XGBoost`.
No LightGBM63, seed averaging or specialist was added.

- Seed 42: standalone XGBoost AUC `0.9572204235205716`; candidate
  AUC `0.9577888987828738` versus baseline `0.9577942088132286`;
  gain `-0.00000531003035486588`. Best iteration index/count `730/731`,
  actual boosted rounds `831`; measured fit call `37.236` seconds.
- Seed 2024: standalone AUC `0.9575214242911765`; candidate
  AUC `0.9581790036051865` versus baseline `0.9581967799255812`;
  gain `-0.000017776320394680134`. Best iteration index/count `772/773`,
  actual boosted rounds `873`; measured fit call `38.877` seconds.
- Both blend gains are negative: reject this configuration and retain the
  incumbent. The result does not rule out all XGBoost configurations. Reused
  holdouts are screening evidence, not independent confirmation.
- Actual run `84.426` seconds under internal/external limits `1680/1800`;
  process peak RSS `904.2` MB, outputs `20,940,142` bytes. The native fit
  call includes construction of the prepared fit/evaluation frames.
- Independent checks reproduced ID alignment, fixed arithmetic, AUC and
  log loss, and verified source hashes, four input hashes and all 124
  prior-artifact hashes. Standalone log loss uses XGBoost's float32 output;
  the independent check preserved that dtype to reproduce sklearn's path.
  Format, Ruff, mypy and all 89 tests pass, including a regression for the
  CPU distribution's package-metadata name.
- Command: `timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python xgboost_experiment.py --data-dir data --artifacts-dir artifacts`.
  Results: `artifacts/xgboost_cpu/`. No model persistence, additional folds,
  final refit, submission export, upload, commit or push occurred.

## Fixed source-only teacher feature experiment

Tested one added numeric predictor, `source_teacher_probability`, in the
original 31-leaf competition LightGBM. The final candidate remained an equal
blend of the existing slow CatBoost and this student. No LightGBM63, XGBoost,
seed averaging, specialist, source-row augmentation or weight search was added.

Reused `clean_source_rows` unchanged and reproduced its previous report:
129,859 unique source rows remained after excluding all exact 21-feature
matches with competition train/test rows (17 train, 4 test). Matching excluded
IDs and competition labels; near-duplicate/shared-entity exclusion is not
claimed. Existing source-license/provenance uncertainty remains applicable.

The teacher learned only from source labels. A source-only stratified 90/10
split (seed 42; 116,873 fit / 12,986 validation) selected 531 rounds using
AUC early stopping. The source-validation AUC was `0.994591552246396`, not a
competition score. A fixed 531-round teacher then fitted all clean source
rows without an evaluation set or early stopping. Its cached competition-train
probabilities cover all 699,635 training IDs; no competition labels influenced
teacher training or iteration selection. Students used the original outer
42/2024 holdouts and inner seed-42 partitions, with original settings and only
one added numeric feature; categories remained fit-partition-only.

- Seed 42: baseline AUC `0.9577942088132286`; candidate
  `0.9580277365442882`; gain `+0.00023352773105955738`.
  Student standalone AUC `0.957833997000424`, best/actual rounds `545`,
  fit time `22.631` seconds.
- Seed 2024: baseline AUC `0.9581967799255812`; candidate
  `0.9584612650032445`; gain `+0.0002644850776633012`.
  Student standalone AUC `0.9580359362484241`, best/actual rounds `313`,
  fit time `15.502` seconds.
- Both gains pass the frozen `+0.0002`-each-seed screening gate. These
  overlapping, repeatedly reused holdouts are not independent confirmation;
  do not infer a public leaderboard gain or replace the submitted model yet.
- Actual run: `70.903` seconds under internal/external limits `1680/1800`.
  Output: `43,623,019` bytes; observed process peak RSS `978.8` MB.
  Independent checks recomputed cleaning, source-only split fingerprints,
  student membership/ID alignment, teacher-feature cache alignment, blend
  arithmetic and AUC/log loss. Four input and 127 prior-artifact hashes
  remain unchanged. Format, Ruff, mypy and 92 tests pass.
- Command: `timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python source_teacher_experiment.py --data-dir data --artifacts-dir artifacts`.
  Outputs: `artifacts/source_teacher_feature/`. No new dependency, final
  competition-data refit, test prediction/export, upload, commit or push.
  The source-only teacher's full-source fit is part of the authorized
  experiment, not a final competition-model refit.

## Fixed five-fold source-teacher validation

Reused the verified source-only 531-round teacher probability cache and the
original five-fold CatBoost seed-42/LightGBM31 OOF components. Trained only
five new LightGBM31 students with the one appended teacher-probability
feature. No teacher, CatBoost or baseline LightGBM retraining occurred.
Regenerated the same stratified folds (seed 2026) and original fold-local
inner partitions (seed 42), including exact membership fingerprints.

The fixed candidate was `0.5 * cached CatBoost42 + 0.5 * new student`;
its comparator was the original `0.5 * cached CatBoost42 + 0.5 * cached
LightGBM31` baseline, not the earlier two-seed-averaged challenger.

- Fold 1: AUC gain `+0.0003034547102925833`; student best rounds `795`.
- Fold 2: gain `+0.00020930450697009828`; best rounds `299`.
- Fold 3: gain `+0.0002462851715884229`; best rounds `417`.
- Fold 4: gain `+0.00022601720723969265`; best rounds `777`.
- Fold 5: gain `+0.00026121827398406516`; best rounds `533`.
- Mean fold AUC: baseline `0.9588663222264138`, candidate
  `0.9591155782004288`; mean paired gain `+0.00024925597401497245`.
- Pooled OOF AUC: baseline `0.9588603718243917`, candidate
  `0.9591080056460375`; gain `+0.0002476338216458318`.
  Pooled log loss: `0.22722204633160709 -> 0.22594907738935158`.
- Five of five folds improved, each by more than `0.0002`. The frozen gate
  passes: mean gain at least `0.0002`, at least four positive folds, and
  positive pooled gain. These previously used development folds are not an
  untouched final test, and the result does not forecast a public score.
- Real run `123.471` seconds under internal/external limits `1680/1800`.
  Outputs `36,051,363` bytes under `artifacts/source_teacher_cv/`.
  `oof_predictions.npz` contains all 699,635 training rows exactly once,
  with IDs, folds, teacher/CatBoost/student/baseline/candidate probabilities.
- Independent recomputation verified all five fold and pooled AUC/log-loss
  metrics, teacher-cache identity, old OOF equality, exact fold and inner
  membership, per-fold component alignment, and declared blend arithmetic.
  All 131 prior artifacts and four inputs remain byte-identical. Format,
  Ruff, mypy and 95 tests pass.
- Command: `timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python source_teacher_cv.py --data-dir data --artifacts-dir artifacts`.
  At CV closure no final competition refit, test prediction/export, upload,
  commit or push had occurred. The median of the five one-based best counts
  is `533`; the separately authorized export is recorded below.

## Authorized qualified source-teacher export

After explicit local-refit/export approval, rebuilt the source-only teacher at
its recorded 531 rounds and fitted the student on all 699,635 competition
training rows at the median of the five one-based CV counts
`[795, 299, 417, 777, 533] -> 533`. Both fits used fixed counts, no evaluation
set and no early-stopping callbacks. Teacher rows remained the same 129,859
clean external examples; the reproduced teacher probabilities matched all
699,635 cached training predictions exactly (`max_abs_difference=0.0`).

Reused the unchanged 4,740-tree slow CatBoost native model with SHA-256
`85466352ce440da5aea2234b034a55df1b8c7e35c33c2fed1b66ea8377924653`.
The final prediction is `0.5 * existing CatBoost + 0.5 * new student`.
Teacher and student models, separate categorical-level files, original and
extended feature specifications, lineage and metadata are saved under
`artifacts/source_teacher_export/`.

- CSV: `artifacts/source_teacher_export/submission_slowcat_source_teacher_50_50.csv`.
- SHA-256: `25fe7ff6b4b2d0c48d6b8fabc1c1de6d6d101898e8ed8b0ee2e52acab8200247`.
- Exactly 299,844 unique test IDs in original order; columns `id,satisfaction`;
  finite probabilities in `[0,1]`, no index column. CSV size: 8,203,941 bytes.
- Both the runner's fresh subprocess and a separate independent native-model
  verification reproduced the test predictions. Independent CSV comparison
  had maximum difference `0.0`; all teacher-cache training values also matched
  exactly. LightGBM's native model names replace spaces with underscores;
  preparation preserves the recorded original feature order.
- Actual teacher/student fits: `5.121` / `21.393` seconds. Total run:
  `57.118` seconds, under the 28/30-minute limits. Export directory:
  `11,978,069` bytes. All four inputs and 138 prior artifacts remain unchanged,
  including the previously submitted CSV. Format, Ruff, mypy and 104 tests pass.
- Command: `timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python source_teacher_export.py --data-dir data --artifacts-dir artifacts`.
  The local export itself did not establish a leaderboard result.
  At local-export completion no Kaggle upload, commit or push had occurred.
  The old CSV remains unchanged.

After separate explicit submission approval, uploaded the exact verified CSV
once. Kaggle submission `57698563` completed with public ROC AUC `0.95845`,
versus the prior `0.95831` (`+0.00014`). The submissions API confirmed the
filename `submission_slowcat_source_teacher_50_50.csv` and `COMPLETE` status;
competition-list API reads showed rank `2173` before and `1968` after scoring
(the displayed ranks are time-dependent). The CLI reported nine submissions
remaining that day. No private score or final-submission selection was
established. No commit or push occurred.

## Fixed CatBoost teacher-feature experiment

Tested the same frozen source-only teacher probability as one added numeric
CatBoost feature. Retained all slow-CatBoost settings (depth 6, learning rate
0.03, cap 6000, patience 200, L2 3, model seed 42, two threads), original four
nominal predictors, and numeric rating/NaN/zero handling. The existing source
teacher and winning teacher-feature LightGBM were not retrained.

Used the exact original outer seeds 42/2024 and inner seed-42 partitions.
The comparator is now the **new** public-0.95845 configuration:
`0.5 * original slow CatBoost + 0.5 * teacher-feature LightGBM`.
The candidate changes only that CatBoost component; the LightGBM component
is the cached `student_probability`, not its old `baseline_probability` or
its complete blend. Cached holdout teacher features matched the full cache.

- Seed 42: current baseline AUC `0.9580277365442882`; candidate
  `0.9580026837378872`; gain `-0.000025052806400971406`.
  New standalone CatBoost AUC `0.957913910567226`; best/actual count
  `1532` (zero-based best index `1531`); fit `286.186` seconds.
- Seed 2024: baseline `0.9584612650032445`; candidate
  `0.9582670117556917`; gain `-0.00019425324755273365`.
  New standalone CatBoost AUC `0.9581529400492719`; best/actual count
  `1561` (index `1560`); fit `271.661` seconds.
- Both blend gains are negative, so the frozen `+0.0002`-each-seed gate
  fails. Retain the public-0.95845 submission. Although standalone CatBoost
  improved on both splits, that did not improve its mixture with the current
  teacher-feature LightGBM; individual model gains do not necessarily add.
  This is a finding about this fixed configuration, not a proof that all
  teacher-feature CatBoost variants or other mixtures fail.
- Actual run: `571.536` seconds within internal/external limits `2580/2700`.
  Outputs: `26,613,405` bytes under `artifacts/catboost_teacher_feature/`.
  Independent recomputation verified component identity, prediction IDs,
  split fingerprints, frozen teacher features, new-incumbent blend arithmetic,
  AUC/log loss and the failed gate. All 145 prior artifacts and four inputs
  remain unchanged. Format, Ruff, mypy and all 110 tests pass.
- Command: `timeout --signal=TERM --kill-after=30s 2700s .venv/bin/python catboost_teacher_feature.py --data-dir data --artifacts-dir artifacts`.
  No further CV, final refit, model persistence, export, upload, dependency
  addition, commit or push occurred. The two reused holdouts remain screening
  evidence, not independent confirmation.

## Fixed label-free distance-group profile experiment

Public research identified distance-group context as a different information-use
hypothesis. Inspiration: [Koumei Maki's feature ladder](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/747358)
and its [Apache-2.0 notebook, version 3](https://www.kaggle.com/code/koumeimaki/s6e10-feature-ladder-what-helps-lightgbm).
The author reports a broader count/mean/residual block gain of about 0.00124
on reused five-fold development predictions; that is not our measured result.
Our independently implemented smaller block does not copy or execute that
notebook, use its outputs, or assume distance uniquely identifies a real route.

Added exactly 17 numeric predictors to the existing teacher-feature LightGBM31:
per-exact-Flight-Distance row count plus means of the other 16 original numeric
predictors (13 ratings, age and two delays). No categorical-code means,
teacher-probability aggregates, target encoding, residuals or parameter search.
Original inputs, rating-zero/NaN semantics, four native categorical columns,
source-only teacher cache, LightGBM settings and original slow CatBoost stayed
unchanged. The student has 39 predictors: 35 numeric and four categorical.

For each original outer seed 42/2024, fitted the table using **only the 503,737
recorded inner model-fit predictors**, excluding the 55,971 early-stopping rows,
139,927 outer holdout rows, test and source rows. Counts use group size; means
skip numeric NaNs and preserve observed zero. Unknown/missing distance keys get
count zero and model-fit global means; all-NaN columns remain NaN. Category
levels and early stopping remain inside the same recorded inner partition.
The teacher was not refitted and learned no competition labels.

Compared `0.5 * unchanged slow CatBoost + 0.5 * new profile/teacher student`
against the current public-0.95845 configuration, using verified ID-aligned
cached components rather than the obsolete pre-teacher baseline:

- Seed 42: baseline AUC `0.9580277365442882`; candidate
  `0.9593360799332391`; paired gain `+0.0013083433889509166`.
  Student-only AUC `0.9594192451978314`; best/native actual rounds `560`;
  fit `27.137` seconds; fitted table contains `3425` distance groups.
- Seed 2024: baseline `0.9584612650032445`; candidate
  `0.9595753150002255`; paired gain `+0.0011140499969810591`.
  Student-only AUC `0.9594326832310826`; best/native actual rounds `401`;
  fit `21.694` seconds; fitted table contains `3423` distance groups.
- The frozen `+0.0002`-each-seed screening gate passes. These overlapping,
  repeatedly reused holdouts are stability evidence, not untouched confirmation
  or a forecast of the public/private leaderboard gain. Keep the already
  submitted 0.95845 CSV until separately approved stronger validation/export.
- Actual total run `64.020` seconds under graceful/external limits `1680/1800`.
  Outputs `27,027,063` bytes in `artifacts/distance_profile_feature/` include
  two prediction CSVs, two fitted profile NPZ tables, metadata and the protected
  baseline manifest; no model checkpoint or submission was saved.
- Independent verification rebuilt both tables and fallback means directly from
  the recorded inner-fit rows, checked table-membership fingerprints and
  prediction IDs, recomputed both blends and all AUC/log-loss metrics, and
  confirmed all 148 prior artifacts, four inputs and experiment source hashes
  unchanged. Ruff format/lint, mypy and all 118 tests pass.
- Command: `timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python distance_profile_experiment.py --data-dir data --artifacts-dir artifacts`.
  No five-fold profile validation, teacher/CatBoost refit, final full-data refit,
  submission export/upload, project dependency addition, commit or push occurred.

## Fixed five-fold distance-profile validation

After separate approval, trained exactly five new original-configuration
LightGBM31 students with the same 17 distance-group features. Reused the frozen
source-only teacher cache and original CatBoost OOF predictions. The comparator
is the accepted teacher-feature blend (`source_teacher_cv`'s **candidate**),
not its historical pre-teacher baseline. Each profile table and categorical
mapping used only that fold's recorded inner model-fit predictors; no
held-out/early-stopping/test/source rows entered the profile table. No teacher
or CatBoost retraining, weight search or new dependency was required.

Paired blend AUC gains versus the current public-0.95845 configuration:

- Fold 1: baseline `0.9592259550223543`, candidate `0.9605510238223502`,
  gain `+0.0013250687999959432`; selected/native rounds `470`.
- Fold 2: baseline `0.9589768416152670`, candidate `0.9601742974329484`,
  gain `+0.0011974558176813810`; rounds `542`.
- Fold 3: baseline `0.9595216637455891`, candidate `0.9608322048344501`,
  gain `+0.0013105410888609725`; rounds `1056`.
- Fold 4: baseline `0.9587929214165344`, candidate `0.9599223961815564`,
  gain `+0.0011294747650220094`; rounds `414`.
- Fold 5: baseline `0.9590605092023987`, candidate `0.9603313930907451`,
  gain `+0.0012708838883463969`; rounds `736`.

Mean paired gain `+0.0012466848719813405`; all five folds positive.
Pooled OOF baseline `0.9591080056460375`, candidate `0.9603568885695387`,
gain `+0.0012488829235012266`. The unchanged advancement gate passes:
mean gain >=0.0002, at least four positive folds and positive pooled gain.
The median selected count is `542`; it is a possible fixed refit count, not
permission to refit. These reused development folds are stability evidence,
not an untouched test or a promised public/private-score gain.

Actual command: `timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python distance_profile_cv.py --data-dir data --artifacts-dir artifacts`.
Total runtime `186.764` seconds, within graceful/external limits `1680/1800`;
outputs `37,095,944` bytes under `artifacts/distance_profile_cv/`.
Independent read-only verification rebuilt all five tables/global fallbacks
from their recorded inner-fit memberships, checked 39-feature definitions,
source/teacher/CatBoost identities and current-baseline arithmetic, then
recomputed every saved fold and pooled AUC/log loss and the gate. Every one of
699,635 training rows has exactly one OOF prediction. All 154 prior artifacts
and four inputs remain unchanged. Format, Ruff, mypy and all 120 tests pass.
Independent code review also completed against unchanged runner/test SHA-256
fingerprints. Its initial timeout concern assumed an unwrapped invocation;
a focused second review inspected the actual documented external `timeout`
launcher and returned `passed: true`, with no correctness/security blockers.
Additional finalization-interruption test coverage was suggested, not required.
The model fits were not rerun. No final refit, model checkpoint, submission
export/upload, commit or push occurred.

## Authorized distance-profile final export and public result

The user authorized the full sequence: commit/push the existing checkpoint,
local refit/export, one Kaggle upload, then commit/push the implementation and
actual result without intermediate approval pauses unless something breaks.
The first checkpoint was pushed as `fdb111eb3cda5b1900c3a5274cb4a6b5da69e633`.
Data, artifacts, environments and credentials remain excluded from Git.

Implemented `distance_profile_export.py` with focused synthetic tests. Parent
review found the native LightGBM space-to-underscore feature-name convention;
a real native fixture first reproduced the rejected-name guard, then verified
the targeted fix while preserving original inference names/order. A separate
read-only review prompted an explicit fresh-reload subprocess timeout and a
real-child termination regression. Focused re-review checked the unchanged
final source hashes and verified CPython direct-child cleanup plus the GNU
external-timeout boundary; it returned `passed: true` with no blockers.
All 127 tests, Ruff format/lint, mypy and whitespace checks pass.

Ran `timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python distance_profile_export.py --data-dir data --artifacts-dir artifacts`.
Only the new LightGBM student was fitted: exactly `542` rounds on all `699635`
competition-training rows, without evaluation sets, callbacks or early stopping.
Fitted the profile table on all original training predictors only: `3474`
exact-distance groups and the same 17 added features, with unchanged zero/NaN
and unknown-distance policies. Reused the native source-only teacher (`531`
rounds) and original slow CatBoost (`4740` trees), without refitting either.
The teacher reproduced its complete cached training probability vector within
the declared `1e-12` tolerance. The final blend remains exactly 50/50.

Exported `artifacts/distance_profile_export/submission_slowcat_teacher_distance_profile_50_50.csv`:

- SHA-256: `048abb59dc3ffb20e495dd414f7e7be3ca0e795d3fa66145516957c4b1d148e7`.
- Exactly `299844` rows, columns `id,satisfaction`, original test-ID order,
  unique IDs, no index and finite probabilities in `[0,1]`.
- Independently rebuilt the full-training profile counts, means and global
  fallbacks; reconstructed native teacher/student/CatBoost predictions in a
  separate process. Maximum absolute native-to-CSV difference was
  `2.220446049250313e-16`. Internal fresh-process component checks also passed.
- All `166` prior artifacts and four CSV inputs remain unchanged. The previous
  public-best CSV retains SHA-256
  `25fe7ff6b4b2d0c48d6b8fabc1c1de6d6d101898e8ed8b0ee2e52acab8200247`.
- Actual export runtime `70.897` seconds under graceful/external limits
  `1680/1800`; output directory size `10414874` bytes. Models, table, feature
  definitions and export metadata are local-only under the ignored artifact
  directory.

After checking the exact CSV hash/schema again, queried live competition state
and submission history. Kaggle reported entered status and deadline
`2026-10-31 23:59:00` UTC, with no matching prior filename. The official rules
had been read live during the immediately preceding public-research review;
this export uses our existing data/models and independently implemented
features, not downloaded public prediction libraries or model weights.
Submitted this verified file **once** via the isolated Python-3.12 Kaggle-2.2.5
CLI. Kaggle returned submission `57726880` and nine submissions remaining
that day. Read-back status: **COMPLETE**, public ROC AUC **0.95982**, an actual
improvement of **+0.00137** over submission `57698563` (`0.95845`).
The competition-list API observed rank `2027` immediately before upload and
`1652` after scoring, with `3323` teams at those observations. These ranks
are time-dependent; no private score or final-submission selection was verified.
The final code/documentation commit follows this recorded public result.

## Fixed source-neighbour probability screen

Approved one final-sprint information-transfer test against the accepted
public-0.95982 distance-profile configuration. Reused the existing
label-independent source deduplication/overlap exclusion policy: `129859`
clean source rows, with no competition target available to the neighbour
feature helper. For each competition row, considered all cleaned source rows
with the same exact Flight Distance. The new probability is the source-label
average weighted by `exp(-d)`, where `d` counts unequal values among the other
20 original predictors. Numeric equality is exact (no rounding); two missing
numeric values agree, observed zero remains valid, and categorical missing
values use the same explicit token. Missing/unseen distance keys use the
cleaned-source positive prevalence `0.43437882626541097`. Alpha is fixed at
one; no neighbour-count, smoothing or weight search was run.

Chunked both query/reference work to at most `250000` pair elements, without a
full query/reference/20-feature tensor. Appended this single numeric feature
to the unchanged 39-feature teacher/profile student, producing 40 predictors
(36 numeric, four nominal). Teacher, original slow CatBoost, LightGBM31
settings, original inner/outer partitions and 50/50 ensemble weights stayed
fixed. Reused each seed's verified profile table fitted only on its inner
model-fit predictors. The comparator is the saved distance-profile
**candidate_probability**, not its obsolete baseline column.

Real measurement:

- Seed 42: baseline AUC `0.9593360799332391`; candidate
  `0.9593987480585240`; gain `+0.00006266812528488153`.
  Student-only AUC `0.9593846704046880`; best/native rounds `911`;
  fit `41.262` seconds.
- Seed 2024: baseline `0.9595753150002255`; candidate
  `0.9596805647640434`; gain `+0.00010524976381787177`.
  Student-only AUC `0.9595619786730625`; best/native rounds `448`;
  fit `24.214` seconds.
- Both local gains are positive but below the frozen `+0.0002`-each-seed
  threshold. Retain the accepted public-0.95982 submission. No five-fold
  neighbour validation, final refit, export or upload followed this failed
  screen; the gate was not relaxed. Reused holdouts remain selection/stability
  evidence, not independent confirmation.

Command: `timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python source_neighbor_experiment.py --data-dir data --artifacts-dir artifacts`.
Actual total runtime `95.506` seconds within graceful/external limits
`1680/1800`. Outputs `33960425` bytes under `artifacts/source_neighbor_feature/`:
one frozen ID-indexed source-neighbour training cache, two holdout prediction
CSVs and comparison metadata, without model checkpoints.

Parent review added failing regressions before fixing query-step skips when
the pair cap is smaller than the requested block, NaN-aware cached profile
comparison, incomplete reused dependency fingerprints and loss of the primary
exception during later integrity failures. A separate all-positive source
fixture reproduced weighted-mean roundoff slightly above one; clipping only
that mathematically bounded numerical result fixed the strict probability
check. Independent code re-review passed against the frozen final source
hashes. Format, Ruff, mypy and all 137 tests pass.

Read-only verification checked all cached IDs/probability ranges, splits,
profile memberships, blend arithmetic and saved AUC/log-loss metrics. An
independent scalar/pandas oracle covered `4487` label-independently selected
rows spanning all observed distance keys plus random rows; its maximum
probability difference was `5.551115123125783e-16`. All 171 prior artifacts and
four input CSV hashes remain unchanged, including the previously submitted
profile CSV SHA-256 `048abb59dc3ffb20e495dd414f7e7be3ca0e795d3fa66145516957c4b1d148e7`.
The screen saved source/tests/docs without a model refit, export or upload.
The user subsequently authorized committing and pushing this completed screen;
the failed advancement decision remains unchanged.

## Fixed distance-group residual screen

Approved one fixed block of 16 numeric features:
`original passenger value - mapped distance-group mean`, for the original
numeric predictors other than Flight Distance. Retained the original 39
teacher/profile predictors; did not add the rejected source-neighbour feature.
The new student has 55 predictors (51 numeric, four nominal). Reused each
seed's verified profile table fitted only on the recorded inner model-fit
predictors, the frozen source-only teacher and original slow CatBoost. No
new aggregation, target encoding, settings or ensemble-weight search occurred.
Numeric NaNs, observed zero and the existing unseen-distance global-mean
fallback remain unchanged.

Compared the complete fixed 50/50 blend against the accepted public-0.95982
configuration on the same original splits:

- Seed 42: baseline AUC `0.9593360799332391`, candidate
  `0.9592117485346789`, gain `-0.0001243313985601624`.
  Student-only AUC `0.9592748841148006`; selected/native rounds `392`;
  fit `29.180` seconds.
- Seed 2024: baseline `0.9595753150002255`, candidate
  `0.9595421606779668`, gain `-0.00003315432225869941`.
  Student-only AUC `0.9593470746406878`; rounds `449`;
  fit `31.922` seconds.
- Both gains are negative, so the frozen `+0.0002`-each-seed gate fails.
  Retain the public-0.95982 submission. No residual five-fold validation,
  full-data refit, export or upload followed. These reused holdouts remain
  diagnostic selection/stability evidence, not independent confirmation.

Supported invocation is from the project root with relative arguments:
`timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python distance_residual_experiment.py --data-dir data --artifacts-dir artifacts`.
Actual total measurement `90.208` seconds under graceful/external limits
`1680/1800`, with `26651755` output bytes under
`artifacts/distance_residual_feature/`. Only two held-out prediction CSVs,
metadata and a protected-artifact manifest were saved, not model checkpoints.

Parent preflight checked actual cached-profile identities and residual
arithmetic before fitting. Independent read-only verification rebuilt the
profile tables from recorded inner-fit rows, checked all original feature
values remained unchanged and residual arithmetic on 1,024 held-out rows,
then recomputed both baseline/candidate/standalone AUC and log loss, own-ID
alignment and blend/gate arithmetic. All 175 prior artifacts and four inputs
remain unchanged, including the accepted submission CSV. Independent code
review passed against the frozen runner/test hashes; its nonblocking notes
include the required project-cwd relative launcher and possible additional
provenance/test coverage. Format, Ruff, mypy and all 140 tests pass.
Source/tests/docs for this screen remain uncommitted; no commit/push was
included in this approval.

## Three-experiment final sprint

User authorized up to **three new experiments**, stopping at the first to gain
at least `+0.0002` AUC on **both** original holdouts, otherwise stopping after
three. Predeclared independent order: categorical Flight Distance duplicate,
127-leaf capacity, then fold-safe route target encoding. Each starts from the
accepted 39-feature teacher/profile configuration with the original slow
CatBoost and fixed 50/50 blend; rejected residual/neighbour features are absent.
Each has graceful/external limits `1680/1800` seconds, sequential fits and two
threads. No new dependencies, parameter/weight search, CV/refit/export/upload,
or commit/push were authorized. All three completed; **none passed**, so stop.

1. **Categorical Flight Distance duplicate:** retain numeric distance and all
   39 existing predictors; append one canonical string category, with its native
   category vocabulary learned only on inner-fit rows. Model has 40 predictors
   and five categorical columns; original 31-leaf settings unchanged.
   - Seed 42 AUC `0.9593017767357015`, gain `-0.000034303197537610686`;
     selected/native rounds `188`, fit `17.326` seconds.
   - Seed 2024 AUC `0.959614366413208`, gain `+0.000039051412982438194`;
     rounds `153`, fit `15.345` seconds.
   - Total `49.846` seconds, `26618282` output bytes; 179 prior artifacts and
     four inputs unchanged. Saved under `artifacts/final_sprint_distance_category/`.
2. **127 leaves:** change only `num_leaves: 31 -> 127`; retain the accepted
   39-feature representation, four categories and all other model settings.
   - Seed 42 AUC `0.959381044370554`, gain `+0.00004496443731494271`;
     rounds `239`, fit `21.349` seconds.
   - Seed 2024 AUC `0.9596189288055259`, gain `+0.00004361380530037984`;
     rounds `266`, fit `21.953` seconds.
   - Total `57.125` seconds, `26617756` bytes; 182 prior artifacts and four
     inputs unchanged. Saved under `artifacts/final_sprint_leaves127/`.
3. **Route target encoding:** append two numeric features for exact Flight
   Distance and Flight Distance × Class. Existing sklearn 1.5.2 TargetEncoder
   uses binary target, smoothing `20.0`, five shuffled stratified inner cross-fit
   folds and seed `42`. `fit_transform` receives **only inner model-fit rows and
   labels**. Early-stopping and outer-holdout rows use `transform` from the
   encoder fitted on all inner-fit rows. Unseen keys receive inner-fit prevalence;
   missing keys are an explicit category. Collision-safe JSON pair keys preserve
   the original predictor values. Model has 41 predictors, four categories and
   the unchanged 31-leaf settings.
   - Seed 42 AUC `0.959470414120363`, gain `+0.00013433418712394118`;
     rounds `433`, fit `23.767` seconds.
   - Seed 2024 AUC `0.9597588993515919`, gain `+0.00018358435136633933`;
     rounds `499`, fit `25.621` seconds.
   - Total `73.038` seconds, `26621581` bytes; 185 prior artifacts and four
     inputs unchanged. Saved under `artifacts/final_sprint_route_target_encoding/`.

The two unchanged baseline AUCs are `0.9593360799332391` and
`0.9595753150002255`. The strongest trial improved both but still missed the
unchanged practical gate. At that three-trial checkpoint, no fourth
configuration or combination was authorized. Keep public ROC AUC `0.95982`;
local holdouts remain reused model-selection data, not untouched confirmation.

One fixed-choice runner, `final_sprint_experiments.py`, and focused
`test_final_sprint_experiments.py` cover all three, reusing the existing cache,
profile, split, model, baseline and gate helpers. Supported project-root command:
`timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python final_sprint_experiments.py --experiment distance_category --data-dir data --artifacts-dir artifacts`;
substitute the next declared choice only after verifying rejection of the prior.
Parent preflight exercised actual teacher/profile caches and incumbent AUCs.
Independent read-only code review passed against runner SHA-256
`06fb5c013978a580ffaf356587bf55db0404b0892e84b14614d660310663c9fb`
and test SHA-256
`48d7a979af637893497000f5fc05c634131d088661f05fb3ad7bce65aa3e1fc8`.
Synthetic smoothed-count oracle checks the encoding's actual complementary
inner folds, original schema/value preservation and unseen-key fallback.
Read-only post-run verification recomputed all six saved baseline/candidate/
standalone AUC and log-loss comparisons, own-ID alignment, blend and gate
arithmetic, actual parameter dictionaries, schemas and input/source/protected
artifact hashes. The public submission CSV remains byte-identical. Format,
Ruff, mypy and all 144 tests pass. New runner/tests and existing docs/residual
work remain uncommitted; no further experiment or submission was performed.

## Two additionally authorized, lesson-driven follow-ups

After the three-trial stop, the user explicitly authorized **two more** trials,
with the same stop-first-pass rule and `+0.0002` on both original holdouts.
Both compare against the **unchanged accepted 39-feature profile blend**, not
against an easier or superseded baseline. Both completed; **neither qualified**.
No third follow-up, new combination, confirmation CV, refit/export/upload or
commit/push followed. Existing submitted public ROC AUC `0.95982` stays selected.

1. `route_target_encoding127`: combine the strongest earlier feature block
   (two route target encodings) with 127 leaves. All other 41-feature student,
   encoding, validation and fixed 50/50 settings remain unchanged.
   - Seed 42 AUC `0.9595202482410241`, gain `+0.00018416830778500426`;
     selected/native rounds `242`, fit `23.162` seconds.
   - Seed 2024 AUC `0.9597192815712393`, gain `+0.00014396657101378718`;
     rounds `188`, fit `19.765` seconds.
   - Total `66.882` seconds; `26622228` output bytes; all 188 prior artifacts
     and four inputs unchanged. Saved in
     `artifacts/final_sprint_route_target_encoding127/`.
   - Capacity helped seed 42 but hurt seed 2024 compared with the preceding
     31-leaf two-encoding trial. Those gains do not add consistently.
2. `route_context_target_encoding`: retain the simpler 31-leaf student and
   extend the two-encoding block with exact Flight Distance × Type of Travel
   and Flight Distance × Customer Type. Four target features total; 43 student
   predictors (39 numeric, four categorical). The first two encodings stay
   bit-identical, with no rejected categorical-distance/residual/neighbour block.
   - Seed 42 AUC `0.9595058750911692`, gain `+0.00016979515793014155`;
     rounds `493`, fit `26.827` seconds.
   - Seed 2024 AUC `0.959814917161006`, gain `+0.00023960216078044994`;
     rounds `546`, fit `29.833` seconds.
   - Total `89.511` seconds; `26623708` bytes; all 191 prior artifacts and four
     inputs unchanged. Saved in
     `artifacts/final_sprint_route_context_target_encoding/`.
   - Only seed 2024 clears the practical gate; a favorable average or one
     qualifying split does not meet the required both-seed rule.

Both use fixed smoothing `20.0`, five shuffled stratified encoding folds and
seed `42`. Only inner model-fit labels enter `fit_transform`; early-stopping
and outer-holdout rows use `transform` from the encoder fitted on all inner-fit
rows. Original teacher/profile features, source-only teacher, original CatBoost,
weights and split rows stay frozen. No extra parameter or key search occurred.
Each trial used the original `1680/1800` graceful/external deadline and two
threads. Parent runs the supported project-root timeout command with each
new `--experiment` choice; the runner never launches the next trial itself.

The existing runner/test received a small additive extension, not another
experiment framework. Historical three-trial metadata and predictions remain
byte-identical and retain their original source SHA. A scratch snapshot of that
old runner (`06fb5c013978a580ffaf356587bf55db0404b0892e84b14614d660310663c9fb`)
was used to check old/new default encoding equality directly. Independent
read-only review passed for the extended runner SHA
`f0ca23087516bd8a2814ba9df71a04879ef4ea915362992bbf509be96be0f958`
and test SHA `6d93e3cbcc8242bf85294a37d62463d488af6326bb620025329b828f0e9f404a`.
Synthetic checks include exact parameter differences, unchanged original
features/specification, cross-fit count oracles, unseen-key fallback and
held-out-label independence. Read-only real-output verification recomputed
all four fitted students' AUC/log loss, baseline/candidate metrics, own-ID and
split alignment, blend/gate arithmetic, actual 31/127-leaf settings, feature
schemas and before/after source/input/protected-artifact hashes. Public CSV
bytes remain unchanged. Format, Ruff, mypy and all 146 tests pass. Source/tests,
preceding residual work and updated existing docs remain uncommitted/unpushed.
Reused holdouts remain exploratory selection evidence, not independent proof.

## User-authorized up-to-20 goal loop

User authorized up to **20 new attempts**, stopping at the **first** gain of
at least `+0.0002` on **both** existing holdouts; after verification, commit and
push the code and stop. This authorizes no Kaggle upload, full-data export,
confirmation CV or new dependencies. Every tested configuration, including a
cached-prediction combination, counts once. All compare against the unchanged
public-0.95982 profile configuration (AUCs `0.9593360799332391` and
`0.9595753150002255`), never a weaker baseline. The reused holdouts remain
selection data; passing this practical gate is not independent confirmation.

First three new attempts, in declared order:

1. **Four context encodings + 127 leaves:** AUCs `0.9595138561737755` /
   `0.9597576674973888`; gains `+0.00017777624053638696` /
   `+0.00018235249716325175`. Reject. Selected/native rounds `189/249`, total
   `78.035` seconds, `26628363` bytes; 194 prior artifacts/four inputs intact.
   Output `artifacts/final_sprint_route_context_target_encoding127/`.
2. **Hierarchical route smoothing:** original 39 predictors plus four route
   means, with 31 leaves. Exact Distance uses the encoding-fit global prevalence;
   Distance×Class/Travel/Customer use that category's encoding-fit prevalence
   as the smoothing prior. Fixed smoothing `20`, five shuffled stratified
   complementary encoding folds, seed `42`; unknown pairs fall back to their
   parent-category prior, unknown parent categories to the global prior.
   AUCs `0.9595073219456269` / `0.9598797365304188`; gains
   `+0.00017124201238782888` / `+0.00030442153019327733`. Reject: seed 42 misses.
   Rounds `609/614`, total `103.491` seconds, `26629964` bytes; 197 prior
   artifacts/four inputs intact. Output
   `artifacts/final_sprint_route_hierarchical_encoding/`.
3. **Source-anchored teacher-residual route encoding:** original 39 plus four
   route/context means of `label - frozen teacher probability`, with 31 leaves.
   Fixed smoothing `20` toward each encoding-fit mean residual; same five
   stratified folds and seed. Negative feature values are valid and checked
   in `[-1,1]`; no teacher refit. AUCs `0.9594541270750733` /
   `0.9598274989179776`; gains `+0.00011804714183416554` /
   `+0.00025218391775205795`. Reject: seed 42 misses. Rounds `266/504`, total
   `84.562` seconds, `26630658` bytes; 200 prior artifacts/four inputs intact.
   Output `artifacts/final_sprint_route_teacher_residual_encoding/`.

All label-derived training features use only complementary encoding-fit rows;
only inner model-fit labels enter any learned statistic. Early-stopping/outer
rows use maps fitted on all inner-fit rows. Original teacher, profile tables,
CatBoost, 50/50 weights and all original predictors stay fixed for these trials.
Read-only independent rescoring verified both held-out CSVs for each trial,
all metrics/blend/gate arithmetic and source/input/protected fingerprints.
A separate real-data groupby oracle checked each novel four-feature formula on
960 model-fit/early-stopping/holdout rows per seed per method: maximum absolute
error `0.0`, all saved encoding-fold fingerprints matched. No model training
was performed by that verifier.

Independent review found a provenance gap in the inherited flat encoder's
metadata, not a modeling/leakage defect. A failing regression was added, then
explicit fold fingerprints/data/source/timing/prior-policy fields were added;
focused re-review passed. Frozen main-runner SHA
`a4d37fe548c03f3493c022d9bec077cf4e886993509ef99a261aec73035b85e4`, test SHA
`5eb5e9bd3c98a141f618046e2a4180649dbca33f51b0d6b542292a8ade65e724`.
Historical output metadata was not rewritten.

4. **Equal average of complementary route students — QUALIFIED:**
   `0.5 original CatBoost + 0.25 context-encoding LightGBM127 + 0.25 hierarchical
   LightGBM31`. Reused the exact already-measured student predictions, each
   aligned by its own IDs; no new fits or target-statistic refit occurred.
   - Seed 42 baseline `0.9593360799332391`, candidate `0.9595596911655389`,
     gain `+0.00022361123229974744`.
   - Seed 2024 baseline `0.9595753150002255`, candidate `0.959864554491251`,
     gain `+0.0002892394910254392`.
   - Both exceed the unchanged `+0.0002` gate. This is the first pass in this
     freshly authorized loop: **stop at attempt 4 of at most 20**. Attempt 5
     (`0.4 Cat + 0.6 hierarchical31`) was declared but **not run**, and no later
     attempts, confirmation CV, refit/export or upload followed.
   - Scoring `7.902` seconds, zero fits/training seconds, `26577902` output bytes;
     203 prior artifacts and four inputs unchanged. Outputs
     `artifacts/cached_route_blends_route_pair_average/compare_metadata.json`
     and the two own-ID validation CSVs. The original public submission CSV
     remains byte-identical; public ROC AUC is still `0.95982`, not a new score.

`cached_route_blends.py` validates source-code/input/prediction fingerprints,
complete source status, model settings, 43-feature schemas and original split/
inner-fit membership before scoring either fixed combination. A failing
metadata-size regression prompted counting final JSON bytes. Independent review
then found mismatched alternate-path lineage and a possible misleading completed
record after a final write failure. Both were reproduced with failing checks:
canonical source/seed filenames are now mandatory and final metadata is staged
and atomically replaced, with failed state/cleared selection on persistence
errors. Current independent focused re-review passed for source SHA
`5fbb9065909d2de6dc4585428a896c8c94b54310ac77ac7ef27c5dd46ace3fff`
and test SHA `9684cbc353033ffed8ed1263b2d3be439e1a7d1cdd4d40380b1067fe01ea06de`.
All 157 tests, format, Ruff and mypy pass. Read-only independent rescoring of
both written CSVs reproduced baseline/candidate/student AUC and log loss,
component arithmetic, IDs, original probabilities and the exact gate. Hash
checks confirm all protected artifacts and inputs stayed unchanged.

The user explicitly authorized commit/push after this pass. The checkpoint
includes the required existing residual/sprint helper source and tests plus
these cached-blend source/tests and actual-result documentation; raw data,
artifacts, virtual environment and credentials remain ignored. Earlier sections'
"uncommitted" statements describe their then-current checkpoints. This selected
candidate has **only** passed the reused-holdout screen; it is not yet a
five-fold-qualified export or Kaggle submission.

## Direct route-pair refit, export and Kaggle submission

User explicitly authorized fit/export/upload for the selected 50/25/25 blend,
then confirmed continuation. No new five-fold model confirmation or additional
candidate search was performed. Full-data counts were frozen before fitting:
ceil(median(`189,249`)) = **219** context-student trees with 127 leaves;
ceil(median(`609,614`)) = **612** hierarchical-student trees with 31 leaves.
Both fit all **699,635 training rows**, without an eval set, early stopping or
callbacks. Each uses the exact existing five-fold, seed-42, smoothing-20
cross-fit training feature helper. Full-training inference maps never supplied
self-target-encoded training features. Teacher 531 and CatBoost 4740 remained
frozen; profile aggregates used all training predictors only, not test rows.

`route_pair_export.py` / `test_route_pair_export.py` reuse the existing fitting,
profile, teacher and metadata helpers. They persist compact full-training route
maps, two native students, feature/category schemas and own-ID test-component
probabilities. All test-row map values matched the original encoders before
model acceptance. Fresh-interpreter native inference reproduced every component
and the fixed `0.5 Cat + 0.25 context + 0.25 hierarchical` export; independent
post-run verification had maximum difference **0.0** on all **299,844 rows**.
All full-training route-map groups, parent priors and the 3,474-group profile
table were independently rebuilt/checked. CSV headers are exactly
`id,satisfaction`, IDs retain original test order, probabilities are finite and
in `[0,1]`. All **206 prior artifacts and four inputs** stayed byte-identical.
Run completed in **158.834 seconds**, output **26,395,228 bytes**.

New output:
`artifacts/route_pair_export/submission_slowcat_route_context_hierarchical_50_25_25.csv`
SHA-256 `fd3dfdec0c6cdf9ba9b48d60648dde357d2ef04406e60e99fd34437c6326c176`.
Export metadata is `artifacts/route_pair_export/export_metadata.json`.
Independent review passed after a reproduced cleanup regression was fixed:
failed exports now clear submission/reload metadata as well as removing newly
publishable output files. Current reviewed runner SHA
`4d93e301bc5967d2ee43807e4d285a5bd15b3538f76cf775df375b5476484a48`, test SHA
`a80921786311941c0425b492f93341b6bb26168d5c65d261e7837c876fd38829`.
Format, Ruff, mypy and all **165 tests** pass. Reviewed parent preflight also
verified the actual absolute-to-relative cached qualification path and exact
saved component lineage before full-data fits.

The verified exact CSV was uploaded **once**, after live history/deadline/entry
checks found no duplicate. Kaggle API readback confirms submission **57823822**,
**COMPLETE**, public ROC AUC **0.96001**. The preceding **57726880** remains
preserved at **0.95982**; decimal score delta **+0.00019**. Leaderboard API rank
was **1847** immediately before upload and **1796** at the scoring check,
among **3623** teams; this is a time-dependent observation, not a final rank.
Upload returned **9 submissions remaining that day**. No retry, second upload,
private-score claim or final-submission-selection change occurred.

User subsequently authorized commit/push for this export checkpoint. It includes
`route_pair_export.py`, `test_route_pair_export.py` and these two updated docs;
raw data, generated artifacts and the virtual environment remain excluded.

## Second up-to-20 loop against the released route-pair incumbent

The user authorized a fresh maximum of 20 counted configurations, stopping at
first `+0.0002` AUC gain on **each** recorded holdout, then verifying, committing
and pushing. The fixed comparator is now the public-`0.96001` route-pair blend:
seed 42 `0.9595596911655389`, seed 2024 `0.959864554491251`. The lower profile
baseline is not used. Confirmation CV, full-data export and Kaggle upload are
not part of this authorization. Reused overlapping holdouts remain selection
and stability evidence, not independent confirmation or a public-gain promise.

The first five configurations were independently rejected; the sixth passed
(gains 42/2024):

1. `pair_light60`: `40% Cat + 30% context + 30% hierarchical`,
   `+0.0001909405054167923 / +0.0001444296803210232`.
2. `pair_rank_average`: average-tie ranks divided by `N+1`, then `50/25/25`,
   `-0.00008738146304443362 / -0.000032219412584799656`.
3. `pair_logit_average`: clip at `1e-6`, average logits at `50/25/25`, sigmoid,
   `+0.000017290034929118292 / +0.00004427908520310275`.
4. `boarding_route`: append one cross-fitted Distance×Online-boarding target
   mean to **both** current students, with boarding-specific target prevalence
   as prior; retain original `50/25/25` weights.
   `+0.000006469252133323167 / -0.00003770678193737442`.
   Best rounds context/hierarchical: `252/434` and `201/316`.
   Actual `167.738` seconds, four fits; 224 prior artifacts/four inputs unchanged.
5. `teacher_route`: append one Distance×frozen-source-teacher-confidence-bin
   target mean, using bin prevalence as prior. Five fixed width-`0.2` bins,
   `floor(5*p)` capped at 4; unchanged `50/25/25` weights.
   `+0.00012208351812903473 / -0.000001945969582584084`.
   Best rounds context/hierarchical: `191/230` and `157/394`.
   Actual `159.741` seconds, four fits; 231 prior artifacts/four inputs unchanged.

6. **`pair_light70` — first full gate pass; STOP at 6/20.** One fixed counted
   step to `30% original Cat + 35% original context127 + 35% original
   hierarchical31`, reusing the original accepted components, not the rejected
   new-feature students. This followed the predeclared positive `40/30/30`
   near-miss; no other weights were scored in this step.
   - Seed 42: baseline `0.9595596911655389`, candidate `0.9598727586443816`,
     gain **`+0.00031306747884274344`**.
   - Seed 2024: baseline `0.959864554491251`, candidate `0.9600868558567431`,
     gain **`+0.0002223013654921724`**.
   - Both exceed the unchanged `+0.0002` gate against the latest route-pair
     incumbent. Stop immediately; no attempt 7 or later was run.
   - `10.139` seconds, **zero fits/training seconds**, `32228900` output bytes;
     all 238 prior artifacts and four inputs unchanged.
   - Outputs `artifacts/round2_pair_light70/compare_metadata.json` and
     `validation_pair_light70_seed42.csv` / `validation_pair_light70_seed2024.csv`.

The first three and sixth reuse saved original probabilities and perform no fits.
Feature trials retain every original 43-feature value, append only one numeric feature
(44 total, 40 numeric/four categorical), and preserve model parameters:
context 127 leaves, hierarchical 31 leaves, two threads, 1,500-round cap,
learning rate `0.04`, min child size 40, L2 1, AUC-only inner early stopping.
Each extra mean uses smoothing 20 and five shuffled stratified folds, seed 42,
within the **503,737 inner-fit rows only**. Encoding helpers accept only these
labels; early-stopping/outer rows use inner-fit maps. Missing/observed zero are
preserved; unseen keys use their parent prior, unseen parents the fit prevalence.
Frozen teacher, original CatBoost and inner-fit-only profile tables are reused.

`round2_cached_experiments.py` and `round2_feature_experiments.py` reuse existing
source/split/own-ID/cache, metric, gate and atomic-metadata helpers. Independent
review and the then-current 176 tests passed before feature fits. Added failing checks caught a seven-versus-six output
schema mismatch and an incorrect epsilon at teacher-bin boundaries; both were
fixed before real fits. Students are built/fitted sequentially and each finished
component's own-ID probabilities/metadata are persisted immediately. Actual
input/cache/feature preflight passed before training. Independent post-run
rescoring reproduced saved blend arithmetic, metrics, AUC gains, component
NPZ/CSV agreement, model settings and fingerprints. Earlier run source metadata
and all original submitted CSV bytes remain unchanged.

Conditional `40/30/30` reweights of a new-feature trial are allowed only if it is
positive on both seeds but not qualified. Neither measured feature trial meets
that condition, so neither conditional reweight ran. The implemented
`joint_passenger` feature mode is also **not run** at this checkpoint.
The fixed `30/35/35` step passed and closed the loop at attempt **6 of at most
20**. All remaining trials, including `joint_passenger`, remain unmeasured.

`round2_light70_experiment.py` / `test_round2_light70_experiment.py` received
independent review before scoring (runner SHA-256
`785daf65cb8a4de36075231e7da95c5a3030e76239281cb1ebb5ea3c10e71ef1`).
Independent post-run verification joined every component by its own IDs and
recomputed both baseline/candidate AUC and log loss, original-component
arithmetic, unchanged gate and current input/source/protected fingerprints.
Format, Ruff, mypy and all **179 tests** pass. Source and actual-result docs
are authorized for commit/push on this first pass. No further confirmation CV,
full-data refit/export or Kaggle upload occurred; the previously submitted
public-`0.96001` CSV remains byte-identical. These selected holdout gains do
not establish an improvement on the public or final private leaderboard.

## Authorized 30/35/35 export and Kaggle release

After the six-attempt loop stopped, the user explicitly authorized fit/export,
Kaggle upload and another commit/push. The selected change is only weights:
`0.30 CatBoost + 0.35 context127 + 0.35 hierarchical31`. The exact full-data
fits already exist and remain the same: context 219 trees and hierarchical
612 trees on 699,635 rows, frozen source teacher 531 and CatBoost 4,740.
The release **reuses these verified fits, with zero new training**, instead of
repeating identical fits. It does not include rejected boarding/teacher-bin
features, a new experiment or new five-fold model confirmation.

New entry point `route_pair_light70_export.py` reuses the existing CSV writer,
fresh-interpreter native inference, invalidation and atomic-metadata helpers.
The source export's seven checkpoint files, native model hashes, original
submission and all recorded source files are checked before reuse. Winner
validation regenerates the two canonical holdouts, checks exact own row/ID
vectors, recomputes baseline/candidate AUC and log loss from training labels,
and evaluates the unchanged gate on those recomputed AUCs. Independent review
initially found the missing metric-rescoring guard; a forged-metric regression
and the guard closed that issue. A full synthetic export regression also
caught incorrect use of a plain NPY as a context manager before actual export.
Final focused independent review passed for runner SHA-256
`6301abd0d95e0cdd266ed75990c95214bba4c9428cba2542af1f4adeaf0727ff`,
test SHA-256 `fdc1db1ada332b96fa2368ef3c06c15efeec4c65061b689ea9fd48134edcd732`.
Codex quota exhaustion delayed review; no unreviewed upload occurred.
Format, Ruff, mypy and all **184 tests** pass.

The real export finished in **27.545 seconds**, **8,291,489 output bytes**.
Fresh-interpreter inference reproduced all three cached component vectors with
maximum difference **0.0** on all **299,844 test rows**. The fresh components,
not the child's old 50/25/25 blend column, formed the new 30/35/35 scores.
CSV serialization difference was `2.220446049250313e-16`; independent
round-trip CSV verification reproduced the new weighted probabilities exactly
(maximum difference **0.0**). Headers are exactly `id,satisfaction`, test IDs
are unique and retain original order, and probabilities are finite in `[0,1]`.
All **241 prior artifacts and four input files** stayed byte-identical; native
models, old submitted CSVs and source metadata were not rewritten.

New CSV:
`artifacts/route_pair_light70_export/submission_slowcat_route_context_hierarchical_30_35_35.csv`

SHA-256: `9b8bee3c3290ac93472b845f586363b53f805767451845411a99fde4b165acaa`.
Metadata: `artifacts/route_pair_light70_export/export_metadata.json`.
Reproduce from the project root:
`timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python route_pair_light70_export.py --data-dir data --artifacts-dir artifacts`.
Existing output directories are intentionally not overwritten.

Live checks confirmed entered status, the `2026-10-31 23:59 UTC` deadline,
and no matching earlier upload. The exact verified CSV was submitted **once**:
**57831976**, API readback **COMPLETE**, public ROC AUC **0.96027**.
The prior **57823822 / 0.96001** remains preserved; decimal public gain is
**+0.00026**. API account rank was **1808** before upload and **1710** at the
scoring check among **3656** teams; these are time-dependent observations,
not final rankings. Upload returned **8 submissions remaining that day**.
No retry upload, final-submission-selection change or private-score claim was
made. The user authorized this source/test and actual-result documentation
checkpoint for commit/push; data, generated artifacts and environment stay
ignored.

## Third bounded loop after public 0.96027 — stopped after ten rejections

The user authorized a fresh loop with a maximum of ten failed configurations.
On a first full gate pass only, the requested sequence was commit/push, full-data
fit or verified fit reuse, export/upload once, a second commit/push, then stop.
The comparator was frozen at **30/35/35**, public submission **57831976 / 0.96027**:
local AUC **0.9598727586443816** (seed 42) and **0.9600868558567431** (seed 2024).
Every candidate needed an additional **+0.0002 on BOTH** original holdouts.
No obsolete 50/25/25/profile baseline or favorable average replaced that rule.

All ten predeclared configurations completed and were independently rejected.
AUC gains below are seed 42 / seed 2024 (qualification used unrounded values):

1. `light80`, fixed `20% Cat + 40% context + 40% hierarchical`:
   `+0.00004578346619688034 / +0.000010696417874545006`.
2. `agreement_weight`, label-free row routing with
   `wCat=clip(0.20+abs(context-hierarchical),0.20,0.50)` and equal students:
   `+0.000049479649596007036 / +0.000015628180216387122`.
3. `uncertainty_weight`, label-free routing with `m=(context+hierarchical)/2`,
   `wCat=0.20+0.30*4*m*(1-m)`:
   `+0.00004101166073400275 / +0.00001864956493313752`.
4. `joint_passenger`, one nested-cross-fit distance×class×travel×customer target
   mean appended to both students; original 30/35/35 weights:
   `-0.00006322186992568923 / -0.00003594084144209475`.
5. `slow_context`, context learner only, LR `0.04 -> 0.02`, cap `1500 -> 3000`,
   inner AUC patience `100 -> 200`; all other settings/representation unchanged:
   `+0.0000018193284314627078 / -0.000005180699114260001`.
6. `slow_hierarchical`, the same fixed slower-step policy in the hierarchical
   learner only: `-0.00001513734229119379 / -0.000030403395033018477`.
7. `hierarchical63`, only hierarchical leaves `31 -> 63`:
   `+0.000004485828223321775 / -0.00007167371824889912`.
8. `context_l2_5`, only context L2 `1 -> 5`:
   `-0.0000007139084496898818 / -0.000000440553938885202`.
9. `teacher_consensus`, one predictor-only, inner-fit distance-group mean of
   frozen teacher probabilities appended to both students, with inner-fit
   global-mean fallback and no competition labels in that table:
   `+0.000011194291419491798 / -0.00003384422683005983`.
10. `fusion63`, one 63-leaf student sees original 43 context features plus three
    hierarchical pair encodings (46 total; duplicate distance-only mean omitted).
    Blend `0.30 Cat + 0.175 old context + 0.175 old hierarchical + 0.35 fusion`:
    `+0.000028496328287719308 / -0.000005165179365329564`.

**STOP: ten completed configurations, ten gate failures, no qualifier.**
Best worst-seed gain was attempt 3: `+0.00001864956493313752`, still far below
`+0.0002`. These limited fixed tests do not establish an AUC ceiling. No eleventh
attempt, new confirmation CV, full-data refit, export, upload, commit or push
followed. Conditional publication authorization was not triggered. The existing
0.96027 submission, native fits and all earlier artifact/input bytes remain
preserved. Source/tests and this actual-result documentation remain uncommitted.

`round3_experiments.py` / `test_round3_experiments.py` reuse the established
cache, split, teacher/profile, target-encoding, fit, own-ID, metric/gate and atomic
metadata helpers. All fits used the original inner-fit/early-stopping rows,
AUC-only stopping and two threads; outer labels supplied scoring only. The first
three candidates used saved probabilities with zero fits. Remaining candidates
performed **18 sequential diagnostic fits** in total, saving each component
before the next fit. Sum of measured trial wall times was **973.815 seconds**
(excludes coding/review), outputs **358,918,105 bytes**. Last trial verified all
**286 pre-existing artifacts and four inputs** unchanged; subsequent independent
rescoring verified the complete ten-result set against each recorded manifest.
Repeated, overlapping holdouts remain model-selection/stability evidence, not
untouched confirmation or a promise of a public-score change.

Independent review closed metadata/signal findings with failing regressions:
count timeouts as failed while preserving their cause, persist the baseline
submission fingerprint, best-effort persist failed state after final write
failure, and restore each signal handler even when timer cleanup fails.
Final source SHA-256
`340d40fdd033c723bdd6dd735902b2d5e0615bd8c15e8d0de738a0eda8640c26`,
test SHA-256 `e71fd785e87f88fc4a712ab1e6193d6c9b395e078fa92e33b597c92fa7a49f4a`.
Format, Ruff, mypy and **195 tests** pass. Attempts 1–2 retain their historical
source hashes `6d7d9d0bbe5445f0e8c9dbf7d607f6a130087878881a8c6bd5515c0c9991742d`
/ test `f18fc419a54d2beddd3171b440c9d8caa472d46b103633503ecf470c5cf33fb9`.
Only the timeout-flag assignment moved afterward; no feature/model/formula
changed, no historical metadata was rewritten and no completed fit was rerun.
Hash-verified prior source snapshots remain in the active-profile scratch.

Each ignored result is `artifacts/round3_MODE/compare_metadata.json` with its
own-ID `validation_MODE_seed42.csv` / `validation_MODE_seed2024.csv` and native
component NPZ files where applicable. Reproduce one configuration from the root:
`timeout --signal=TERM --kill-after=30s 1800s .venv/bin/python round3_experiments.py --experiment MODE --data-dir data --artifacts-dir artifacts`.
The internal deadline is 1680 seconds for the whole configuration; existing
output directories are deliberately not overwritten. The bounded loop is closed.

## README and preservation closeout

The user subsequently explicitly authorized the final technical README and
commit/push of all pending source/documentation, including the rejected
round-three runner/tests. This is separate from the earlier conditional gate
publication rule. `README.md` documents design, versions, setup, fixed model
configuration, baseline versus warm-cache execution and retention limits.
Data and generated artifacts remain ignored; no deletion or backup was performed.

A read-only Kaggle API check on **2026-11-01 UTC** reported submission
**57831976 COMPLETE**, public AUC **0.96027**, private AUC **0.96117**.
The competition-list API reported account rank **1662 / 3747** after the
2026-10-31 23:59 UTC submission deadline. These are dated observations;
final-submission-selection flags were not read or changed.

Reproducibility audit: no tracked/local notebook exists and the guarded scripts
consume ignored historical split/cache/qualification/model files. Current
requirements are not a complete machine image; saved metadata contains absolute
paths, timing/resource values and historical source fingerprints. A local
recovery test rebuilt the winning 299844-row CSV from the preserved component
NPZ with exact SHA-256 `9b8bee3c3290ac93472b845f586363b53f805767451845411a99fde4b165acaa`.
A tiny synthetic repeat-fit test produced identical predictions but different
CatBoost model GUIDs and serialized file hashes. Thus a Git checkout plus
retraining cannot promise exact regeneration of every artifact. Preserve and
verify `data/`, all `artifacts/` and any needed external source snapshots before
removing local copies. Same-host CSV recovery is not a cross-machine retraining
verification.

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
