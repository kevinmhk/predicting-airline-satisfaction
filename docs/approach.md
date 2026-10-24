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
