# Predicting Airline Satisfaction

This repository contains the Python experiments and guarded export scripts for
the 2026 Kaggle Playground Series airline-satisfaction competition. The target
is the probability of `satisfaction`; the competition metric is ROC AUC.

## Latest verified result

The approved local blend is:

| Component | Configuration | Weight |
| --- | --- | ---: |
| CatBoost | Original 21 predictors, depth 6, learning rate 0.03, 4,740 iterations | 0.30 |
| Context LightGBM | 219 trees, 127 leaves, learning rate 0.04, L2 1, `min_child_samples` 40 | 0.35 |
| Hierarchical LightGBM | 612 trees, 31 leaves, learning rate 0.04, L2 1, `min_child_samples` 40 | 0.35 |

The latest exported file is:

```text
artifacts/route_pair_light70_export/submission_slowcat_route_context_hierarchical_30_35_35.csv
SHA-256: 9b8bee3c3290ac93472b845f586363b53f805767451845411a99fde4b165acaa
```

It contains 299,844 test rows and the header `id,satisfaction`. Kaggle public
submission `57831976` scored public ROC AUC **0.96027**. A read-only Kaggle API
check on **2026-11-01 UTC**, after the submission deadline, reported private
ROC AUC **0.96117** for the same submission and account rank **1662 / 3747**.
The earlier public-rank observation was 1710. These are dated API observations;
final-submission-selection flags were not checked. The submission deadline was
2026-10-31 23:59 UTC; do not assume new uploads remain available.

## Technical design

The competition data has 699,635 training rows and 299,844 test rows. The
external source dataset contains 129,880 rows. After removing 21 exact-feature
overlaps with the competition data (17 training rows and 4 test rows), 129,859
source rows remain. In the released solution, source labels are used only for
the external teacher.

The four categorical predictors are `Gender`, `Customer Type`, `Type of Travel`
and `Class`; the other 17 original predictors are numeric. IDs and target are
excluded from model inputs. Observed rating zeros remain valid values and
numeric missing values are preserved for native tree handling.

The feature pipeline is:

1. Train a source-only, 531-tree LightGBM teacher on the original 21 predictors
   and append one teacher probability.
2. Build 17 label-free distance-profile features: a count and 16 means,
   excluding the raw distance column. This produces 39 base features.
3. Add four route target-mean features for distance, distance/class,
   distance/travel, and distance/customer groups. The student matrix has 43
   features: 39 numeric and 4 categorical.
4. Fit the context and hierarchical LightGBM students and combine their test
   probabilities with the frozen CatBoost component using the weights above.

Training target maps are cross-fitted. In a diagnostic split, mappings and
preprocessing are learned from inner-fit rows; early-stopping and outer rows
use those mappings without their labels. Outer labels supply scoring only.
For final fitting, training encodings stay cross-fitted and inference maps use
all training labels only; test labels are never used. Context maps use
smoothing 20 and five stratified folds with seed 42.
Hierarchical maps use conditional parent priors for the route-pair keys.
Both full-data students fit all 699,635 rows without an evaluation set or
early stopping. Counts 219 and 612 are the rounded-up medians of the two
selected holdout tree counts. LightGBM uses binary objective/AUC metric,
`random_state=42`, `deterministic=True`, `force_col_wise=True`, and `n_jobs=2`.
The full CatBoost fit also has no early stopping; its loss is Logloss,
`l2_leaf_reg=3`, seed 42 and `thread_count=2`. Its 4,740 trees were selected
from the two validation-fit counts, not the test set.

The final `route_pair_light70_export.py` script is a cache-only release step.
It reuses verified full-data fits, launches a fresh interpreter to reload the
components, checks the old and new predictions, and writes the 30/35/35 CSV.
It does not fit a new model. Existing output directories are intentionally
rejected instead of overwritten.

## Repository map

- `route_pair_light70_export.py`: final cache-only 30/35/35 export.
- `route_pair_export.py`: full-fit route-pair export and explicit `--reload`
  contract; its historical blend is 50/25/25.
- `source_teacher_export.py`: qualified source-teacher export.
- `slower_learning.py`: slower-learning CatBoost/LightGBM comparison and
  optional export.
- `train.py`: bounded baseline training and validation entry point.
- `final_sprint_experiments.py`, `round2_feature_experiments.py`, and
  `round2_light70_experiment.py`: fixed diagnostic or cached experiment
  entry points. Each expects its prerequisite artifacts.
- `round3_experiments.py`: the last ten fixed attempts, all rejected against
  the latest 30/35/35 baseline; no additional submission followed.
- `test_*.py`: unit and contract tests.
- `docs/`: competition, data, rules, approach, and plan records.

There is no notebook in this repository. The Python files are historical,
staged CLI experiments and guarded export steps, not one clean-clone,
end-to-end pipeline.

## Install

The verified environment was Linux x86_64 with Python 3.10.22. CPU execution
with two threads is the supported configuration; an NVIDIA GPU is not
required. Non-Linux platforms, other CPU libraries, and other Python versions
are not verified.

Clone the private repository with an authorized GitHub account, then create a
project-local environment. Supply Python 3.10.22 (the `python3.10` executable)
and run commands from the repository root:

```bash
git clone https://github.com/kevinmhk/predicting-airline-satisfaction.git
cd predicting-airline-satisfaction
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes `requirements.txt`, CatBoost, LightGBM,
XGBoost CPU, NumPy, pandas, SciPy, scikit-learn, joblib, Ruff, and mypy with
the versions recorded in those files. The files do not lock every transitive
package, operating-system library, or CPU implementation.

The verified package set included Python 3.10.22, NumPy 1.26.4, pandas 2.2.3,
scikit-learn 1.5.2, LightGBM 4.6.0, CatBoost 1.2.8, XGBoost CPU 3.2.0,
Ruff 0.8.4, and mypy 1.13.0.

## Data and setup

The following files are required but intentionally ignored by Git:

```text
data/train.csv
data/test.csv
data/sample_submission.csv
data/source_airline/data.csv
```

The competition files are from
[playground-series-s6e10](https://www.kaggle.com/competitions/playground-series-s6e10).
The external source is
[arseniyshutko/binary-aviation-satisfaction-129k](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k). Obtain data separately under
the applicable competition rules and licenses. Keep credentials outside this
repository. The training environment does not
install the Kaggle client. Optional acquisition on a **fresh, empty `data/`
tree**, after authentication and required rules acceptance, uses a separately
installed `uv` and Python 3.12:

```bash
mkdir -p data/source_airline
uvx --python 3.12 --from kaggle==2.2.5 kaggle competitions download \
  playground-series-s6e10 --path data --unzip
uvx --python 3.12 --from kaggle==2.2.5 kaggle datasets download \
  arseniyshutko/binary-aviation-satisfaction-129k --path data/source_airline --unzip
sha256sum data/train.csv data/test.csv data/sample_submission.csv data/source_airline/data.csv
```

The CLI syntax was checked; these downloads were not rerun for the README.
Prefer restoring the original inputs. Compare downloaded bytes with the
manifest below and stop on a mismatch; do not normalize the CSVs or silently
replace their expected hashes. A future dataset version may differ.

Verified input SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `data/train.csv` | `2313b65c74e6994f01c2628aae82d2f97aa6f74a81950172ff83864837c98b36` |
| `data/test.csv` | `ff7932747f27c5274de904194ac38b4d088ea4f9bb005ae74f0d619fb893a9b6` |
| `data/sample_submission.csv` | `6165ecc5769962253534c2d840f26aab55dc3cc091d256bb3641bf6608194a71` |
| `data/source_airline/data.csv` | `2ad76465f6115f93d16121891a6b1235acb8007a4ab065cae0dc46b60ad116e3` |

Use the default paths, or pass alternate paths through the CLI flags below.
The scripts create outputs under `artifacts/`, which is also ignored by Git.

## Configuration

There is no `.env`, YAML, or external configuration service. The supported
path configuration is exposed directly by the scripts:

```text
--data-dir DATA_DIR           default: data
--artifacts-dir ARTIFACTS_DIR default: artifacts
```

`route_pair_export.py` additionally supports `--reload`, `--output-dir`,
`--test-path`, and `--output-path`; all three path flags are required with
`--reload`. `slower_learning.py` supports `--export`. The diagnostic scripts
that select an experiment use their source-declared `--experiment` choices.

The modules set BLAS/OpenMP-related thread variables to `2`. The recorded
training configuration uses seeds 42 and 2024 for outer splits, seed 42 for
inner/estimator operations, and two CPU threads. These settings improve
repeatability but do not establish byte identity across machines.

Inspect an installed entry point before running it:

```bash
.venv/bin/python route_pair_light70_export.py --help
.venv/bin/python route_pair_export.py --help
.venv/bin/python source_teacher_export.py --help
```

## Execution

### Fresh-data baseline (not the released model)

With only the verified data restored, the initial baseline can be run in an
isolated output directory:

```bash
.venv/bin/python train.py --data-dir data --artifacts-dir artifacts/baseline_demo
```

This does **not** recreate the final solution or all historical artifacts.
The full experiment chronology and prerequisite references are recorded in
[docs/approach.md](docs/approach.md) and [docs/plan.md](docs/plan.md).

### Final warm-cache export

The final exporter is a guarded warm-cache workflow. It requires the original
validated artifact lineage, including intermediate qualification artifacts, and
it can reject a moved snapshot because its paths and historical hashes are part
of the guard contract. It is not the portable restore workflow.

Use [docs/restore.md](docs/restore.md) for the smaller private archive that
supports final-model inference and exact CSV recovery. The retention tiers are:

- Keep only the final 30/35/35 CSV and its metadata when exact submission
  retrieval is the only requirement.
- Keep the documented restore manifest, `data/test.csv`, and the listed native
  models and feature artifacts for portable final-model inference and CSV
  recovery.
- Keep `data/train.csv`, `data/source_airline/data.csv`, and
  `data/sample_submission.csv` too when future training matters. They do not
  by themselves supply a clean-clone retraining pipeline. Keep the complete
  historical artifact lineage and referenced source snapshots only when the
  original diagnostics or guarded historical exporters must be investigated.

The slim archive deliberately does not preserve every diagnostic or authorize
rerunning the guarded historical exporters. Do not use the slim archive to
claim that all historical artifacts can be retrained.

With the complete original lineage restored:

```bash
.venv/bin/python route_pair_light70_export.py \
  --data-dir data \
  --artifacts-dir artifacts
```

This writes the CSV shown in [Latest verified result](#latest-verified-result)
only when its fixed destination `artifacts/route_pair_light70_export` is absent.
There is no `--output-dir` flag for this exporter. Do not delete the only copy
of an existing result to force a rerun. Use a verified restored work copy with
all prerequisite paths intact and the destination absent. Absolute-path and
historical-hash checks can reject a moved snapshot; migration is not automated.

### Historical full-fit and reload contracts

These commands are source-verified interfaces, not a claim that a clean clone
can execute the entire historical chain without restored intermediate files:

```bash
# Full-fit historical 50/25/25 route-pair export; expects prior artifacts.
.venv/bin/python route_pair_export.py --data-dir data --artifacts-dir artifacts

# Reload saved route-pair components in a fresh Python process.
RELOAD_DIR="${TMPDIR:-$HOME/.cache}/airline-satisfaction"
mkdir -p "$RELOAD_DIR"
.venv/bin/python route_pair_export.py \
  --reload \
  --output-dir artifacts/route_pair_export \
  --test-path data/test.csv \
  --output-path "$RELOAD_DIR/route_pair_reload.npy"
```

Other stage entry points are independently guarded and may require earlier
outputs. Do not treat the following as a complete bootstrap sequence:

```bash
.venv/bin/python source_teacher_export.py --data-dir data --artifacts-dir artifacts
.venv/bin/python slower_learning.py --data-dir data --artifacts-dir artifacts --export
.venv/bin/python train.py --data-dir data --artifacts-dir artifacts
```

## Quality checks

Run the repository gates in this order after installing dependencies. Unit
tests use synthetic fixtures; they do not require the ignored data/artifacts:

```bash
.venv/bin/ruff format --check --exclude 'artifacts/**' .
.venv/bin/ruff check . --output-format=concise
.venv/bin/mypy --check-untyped-defs *.py
.venv/bin/python -m unittest discover -q
git diff --check
```

The recorded gate run completed with 195 tests passing. The latest ten-attempt
loop also completed independent saved-prediction verification and retained the
released model because no attempt gained +0.0002 AUC on both overlapping
holdouts (seeds 42 and 2024). Those holdouts are selection/stability evidence,
not an untouched confirmation set. No competition model was retrained for this
README update; the small recovery/reproducibility checks below are separate.

## Reproducibility limits

**No: all artifacts cannot be rebuilt exactly from a fresh checkout.** This
repository does not contain the ignored competition/source data, native model
files, split arrays, prediction caches, qualification metadata, or every
historical intermediate. It also contains no notebook that provides a
single end-to-end rebuild. The scripts validate and consume prior artifacts;
they are not a clean-clone pipeline.

Pinned direct dependencies do not lock all transitive dependencies, OS/GLIBC,
CPU behavior, or native model serialization. Fixed seeds and deterministic
settings do not guarantee byte-identical retraining across machines. Some
saved feature specifications contain absolute paths under the original
`/home/kevinmhk/...` root. The [portable restore](docs/restore.md) rebases only a
working copy; the archive and its original hashes stay unchanged. Relocated
native inference was tested in a fresh environment on the same VPS and
reproduced the exact submitted CSV. A physically different machine or OS has
not been tested.

A local recovery check rebuilt the final CSV from the preserved
`artifacts/route_pair_export/component_test_predictions.npz` using the existing
30/35/35 blend and CSV writer. All 299,844 rows reproduced the exact submitted
SHA-256 above on this host. This is recovery from saved probabilities, not
retraining from a clean checkout and not a cross-machine test.

A separate tiny synthetic CatBoost repeat-fit check used identical data,
parameters and seed: predictions matched exactly, but `.cbm` hashes and
`model_guid` values differed. JSON records also contain timestamps, elapsed
times, resource measurements and paths. Thus even matching predictions do not
imply identical model/metadata files. Some historical runs also refer to old
uncommitted source snapshots and recovery steps outside the repository.
Only preserving and verifying the original files can guarantee their bytes.

## Retention and deletion warning

**GitHub is not a data or model backup. Before deleting local files, preserve
and verify the private files appropriate to your retention goal.** The
[portable restore guide](docs/restore.md) lists 13 files totalling 60,869,255
bytes (about 61 MB), including test data, native final models, feature maps,
component predictions, and the submitted CSV. It does not require all 1.7 GB
of experiment outputs. Preserve training inputs separately if future training
matters. Preserve the full historical tree and referenced source snapshots only
if full experimental provenance or historical guards matter; the slim archive
does not preserve those capabilities.

Virtual environments and ordinary package caches can usually be recreated.
Ignored data and model artifacts cannot be recreated byte-for-byte from this
repository alone. Do not delete them until the preserved copy has been
verified.
