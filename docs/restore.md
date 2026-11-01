# Portable private artifact restore

This procedure restores the approved final model components on another machine
without changing source code or retraining. It supports final-model inference
and recovery of the approved 30/35/35 submission. It does not provide a
bitwise prediction guarantee across arbitrary operating systems, CPU libraries,
or Python environments.

The portable workflow is separate from the guarded historical exporters. Those
exporters require their complete original lineage, paths, and source hashes.
A slim archive intentionally gives up rerunning every diagnostic and those
guards.

## Install source and dependencies

Use Linux x86-64 and Python 3.10 for the tested configuration. Install Git and
Python through your normal platform process. Clone the private repository with
your own authorized GitHub access; do not put credentials in these commands.

```bash
git clone https://github.com/kevinmhk/predicting-airline-satisfaction.git airline-restore
cd airline-restore
git checkout 4c73e90ba40a0dab9221956c3dd0cc64938ee083
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
```

This commit contains all inference helpers used below. This guide can be read
from a later documentation commit without changing the frozen inference code.
Run the backup steps on the original machine before removing its files. Run the
verification and restore steps in the destination clone after downloading the
private backup directory through your chosen storage service. No storage
provider or download URL has been selected yet.

## Retained files

The archive is a directory with the following payload files. Keep the paths
relative to the archive root exactly as shown.

```text
artifacts/route_pair_light70_export/submission_slowcat_route_context_hierarchical_30_35_35.csv
artifacts/route_pair_light70_export/export_metadata.json

artifacts/route_pair_export/feature_spec.json
artifacts/route_pair_export/distance_profile_table.npz
artifacts/route_pair_export/route_context_maps.json
artifacts/route_pair_export/route_hierarchical_maps.json
artifacts/route_pair_export/context_student_model.txt
artifacts/route_pair_export/hierarchical_student_model.txt
artifacts/route_pair_export/component_test_predictions.npz
artifacts/route_pair_export/export_metadata.json

artifacts/slower_learning/final_model_catboost_slow.cbm
artifacts/source_teacher_export/teacher_model.txt
data/test.csv
```

Keep a `manifest.sha256` beside these files. The manifest must cover every
payload file, including the retained submission file and all metadata files.
Its comment header must record source commit `4c73e90` and the exact dependency
and environment versions used for the archive. Do not put credentials in the
manifest.

Keep the training inputs separately if future training is required:
`data/train.csv`, `data/source_airline/data.csv`, and
`data/sample_submission.csv`. These files are not required for native final
model inference.

## Create and verify a directory backup

Choose the private storage location separately. The repository does not define
or imply a storage URL. The following commands create the payload directory;
replace `BACKUP_DIR` with a user-supplied private local path.

```bash
BACKUP_DIR=/private/path/chosen-by-you/airline-satisfaction-restore
mkdir -p "$BACKUP_DIR/artifacts/route_pair_light70_export"
mkdir -p "$BACKUP_DIR/artifacts/route_pair_export"
mkdir -p "$BACKUP_DIR/artifacts/slower_learning"
mkdir -p "$BACKUP_DIR/artifacts/source_teacher_export"
mkdir -p "$BACKUP_DIR/data"

cp artifacts/route_pair_light70_export/submission_slowcat_route_context_hierarchical_30_35_35.csv \
  "$BACKUP_DIR/artifacts/route_pair_light70_export/"
cp artifacts/route_pair_light70_export/export_metadata.json \
  "$BACKUP_DIR/artifacts/route_pair_light70_export/"
cp artifacts/route_pair_export/feature_spec.json \
  "$BACKUP_DIR/artifacts/route_pair_export/"
cp artifacts/route_pair_export/distance_profile_table.npz \
  "$BACKUP_DIR/artifacts/route_pair_export/"
cp artifacts/route_pair_export/route_context_maps.json \
  "$BACKUP_DIR/artifacts/route_pair_export/"
cp artifacts/route_pair_export/route_hierarchical_maps.json \
  "$BACKUP_DIR/artifacts/route_pair_export/"
cp artifacts/route_pair_export/context_student_model.txt \
  "$BACKUP_DIR/artifacts/route_pair_export/"
cp artifacts/route_pair_export/hierarchical_student_model.txt \
  "$BACKUP_DIR/artifacts/route_pair_export/"
cp artifacts/route_pair_export/component_test_predictions.npz \
  "$BACKUP_DIR/artifacts/route_pair_export/"
cp artifacts/route_pair_export/export_metadata.json \
  "$BACKUP_DIR/artifacts/route_pair_export/"
cp artifacts/slower_learning/final_model_catboost_slow.cbm \
  "$BACKUP_DIR/artifacts/slower_learning/"
cp artifacts/source_teacher_export/teacher_model.txt \
  "$BACKUP_DIR/artifacts/source_teacher_export/"
cp data/test.csv "$BACKUP_DIR/data/"
```

Generate the manifest with the standard library. The manifest excludes only
itself; every other file in the backup directory is a checksummed payload. The
header is part of the manifest record but is not a payload hash entry.

```bash
.venv/bin/python - "$BACKUP_DIR" <<'PY'
import hashlib
import importlib.metadata
import platform
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root / "manifest.sha256"
files = sorted(
    path for path in root.rglob("*") if path.is_file() and path != manifest
)
if not files:
    raise SystemExit("no payload files found")
distributions = sorted({
    distribution.metadata["Name"]
    for distribution in importlib.metadata.distributions()
    if distribution.metadata["Name"]
})
with manifest.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write("# source_commit: 4c73e90ba40a0dab9221956c3dd0cc64938ee083\n")
    handle.write(f"# python: {platform.python_version()}\n")
    handle.write(f"# platform: {platform.platform()}\n")
    handle.write(f"# machine: {platform.machine()}\n")
    for distribution in distributions:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise SystemExit(f"missing required distribution: {distribution}") from error
        handle.write(f"# package: {distribution}=={version}\n")
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"{digest}  {path.relative_to(root).as_posix()}\n")
print(f"manifest written for {len(files)} payload files")
PY
```

On the destination machine, verify the private directory before copying or
opening any model. This check rejects absolute paths and paths that escape the
archive root.

```bash
ARCHIVE_DIR=/private/path/supplied-by-you/airline-satisfaction-restore
.venv/bin/python - "$ARCHIVE_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root / "manifest.sha256"
if not manifest.is_file():
    raise SystemExit(f"missing manifest: {manifest}")
seen = set()
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    if line.startswith("#"):
        continue
    try:
        expected, relative = line.split("  ", 1)
    except ValueError as error:
        raise SystemExit(f"invalid manifest line: {line!r}") from error
    relative_path = Path(relative)
    if relative in seen:
        raise SystemExit(f"duplicate manifest path: {relative}")
    seen.add(relative)
    if relative_path.is_absolute():
        raise SystemExit(f"absolute manifest path: {relative}")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise SystemExit(f"manifest path escapes archive: {relative}") from error
    if not path.is_file():
        raise SystemExit(f"missing payload: {relative}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"checksum mismatch: {relative}")
if not seen:
    raise SystemExit("manifest contains no payload files")
actual_files = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*") if path.is_file() and path != manifest
}
if actual_files != seen:
    raise SystemExit("manifest does not cover exactly all payload files")
print(f"verified {len(seen)} payload files")
PY
```

The manifest detects file corruption; it does not authenticate an untrusted
download. Obtain it through your approved private storage and retain its own
SHA-256 separately. Never load untrusted model files.

## Restore into a new clone

Use a new clone and a new restore work directory. Copy only the retained
`artifacts/` tree and `data/test.csv`; do not copy source files from the
private archive over the clone.

```bash
WORK_DIR=/path/to/new/clone
ARCHIVE_DIR=/private/path/supplied-by-you/airline-satisfaction-restore
test -d "$WORK_DIR"
test -d "$ARCHIVE_DIR"
mkdir -p "$WORK_DIR/artifacts" "$WORK_DIR/data"
cp -a "$ARCHIVE_DIR/artifacts/." "$WORK_DIR/artifacts/"
cp "$ARCHIVE_DIR/data/test.csv" "$WORK_DIR/data/test.csv"
cd "$WORK_DIR"
```

Use the Python 3.10 environment installed above. The documented quality-gate
commands use `requirements-dev.txt`. Do not put credentials in the archive.

## Rebase the native reload copy

`route_pair_export --reload` reads these four absolute paths from
`artifacts/route_pair_export/feature_spec.json`:
`teacher_model_path`, `context_model_path`, `hierarchical_model_path`, and
`catboost_model_path`. Rebase only those four fields in the copied feature
spec. Run this command only in the new restore work directory, never in the
private archive.

```bash
.venv/bin/python - "$WORK_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
spec_path = root / "artifacts/route_pair_export/feature_spec.json"
record = json.loads(spec_path.read_text(encoding="utf-8"))
replacement = {
    "teacher_model_path": root / "artifacts/source_teacher_export/teacher_model.txt",
    "context_model_path": root / "artifacts/route_pair_export/context_student_model.txt",
    "hierarchical_model_path": root / "artifacts/route_pair_export/hierarchical_student_model.txt",
    "catboost_model_path": root / "artifacts/slower_learning/final_model_catboost_slow.cbm",
}
for name, path in replacement.items():
    if name not in record:
        raise SystemExit(f"missing feature-spec field: {name}")
    if not path.is_file():
        raise SystemExit(f"missing model file: {path}")
    record[name] = str(path)
spec_path.write_text(
    json.dumps(record, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(f"rebased four model paths in {spec_path}")
PY
```

This changes the copied `feature_spec.json`, so its original archive checksum
no longer applies to the working copy. The archive manifest remains the source
of truth for the untouched private payload.

## Recompute and recover the final CSV

Run the native reload in a fresh Python process. The output has exactly four
columns in this order: `catboost`, `context`, `hierarchical`, and the old
50/25/25 candidate. The fourth column is not the approved final blend.

```bash
.venv/bin/python route_pair_export.py \
  --reload \
  --output-dir artifacts/route_pair_export \
  --test-path data/test.csv \
  --output-path restore_reload_predictions.npy
```

Validate the shape and test IDs through the cached component file before
writing a submission. This prints metadata only and never prints data rows.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd

reload_values = np.asarray(np.load("restore_reload_predictions.npy"), dtype=np.float64)
test_ids = pd.read_csv("data/test.csv")["id"].to_numpy(dtype=np.int64)
with np.load("artifacts/route_pair_export/component_test_predictions.npz") as saved:
    component_ids = np.asarray(saved["id"], dtype=np.int64)
if reload_values.shape != (len(test_ids), 4):
    raise SystemExit(f"unexpected reload shape: {reload_values.shape}")
if not np.array_equal(component_ids, test_ids):
    raise SystemExit("cached component IDs do not match data/test.csv order")
if len(test_ids) != 299844 or len(np.unique(test_ids)) != len(test_ids):
    raise SystemExit("unexpected test ID count or duplicate IDs")
if not np.isfinite(reload_values).all():
    raise SystemExit("native reload returned nonfinite values")
print(f"reload shape and test ID order verified: {len(test_ids)} rows")
PY
```

Feed only the first three columns to `round2_light70_experiment.light70_blend`
and use `route_pair_export._write_submission`. The helper applies weights
0.30, 0.35, and 0.35, and the writer validates the `id,satisfaction` schema
and test ID order.

```bash
mkdir -p restore-output
.venv/bin/python - <<'PY'
from pathlib import Path

import numpy as np
import pandas as pd

import route_pair_export as route
from round2_light70_experiment import light70_blend

test_ids = pd.read_csv("data/test.csv")["id"].to_numpy(dtype=np.int64)
reloaded = np.asarray(np.load("restore_reload_predictions.npy"), dtype=np.float64)
if reloaded.shape != (len(test_ids), 4):
    raise SystemExit(f"unexpected reload shape: {reloaded.shape}")
values = light70_blend(reloaded[:, 0], reloaded[:, 1], reloaded[:, 2])
result = route._write_submission(
    Path("restore-output/submission_slowcat_route_context_hierarchical_30_35_35.csv"),
    test_ids,
    values,
)
expected = "9b8bee3c3290ac93472b845f586363b53f805767451845411a99fde4b165acaa"
if result["sha256"] != expected:
    raise SystemExit(f"unexpected recovered CSV checksum: {result['sha256']}")
print(f"recovered {result['rows']} rows; sha256={result['sha256']}")
PY
```

The expected recovered file has 299,844 rows, columns `id,satisfaction`, and
SHA-256
`9b8bee3c3290ac93472b845f586363b53f805767451845411a99fde4b165acaa`.
Treat a mismatch as a failure. Do not upload or overwrite an existing
submission automatically.

## Cached-component fallback

If native model reload is unavailable, the cached component file can recover
the approved CSV without fitting. This is a fallback only. It must validate
the component-file hash recorded by `route_pair_export/export_metadata.json`
and must validate cached IDs against `data/test.csv` before applying the
30/35/35 blend.

```bash
mkdir -p restore-output
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import route_pair_export as route
from round2_light70_experiment import light70_blend

component_path = Path("artifacts/route_pair_export/component_test_predictions.npz")
metadata = json.loads(
    Path("artifacts/route_pair_export/export_metadata.json").read_text(
        encoding="utf-8"
    )
)
expected_hash = metadata.get("files", {}).get(component_path.name)
actual_hash = hashlib.sha256(component_path.read_bytes()).hexdigest()
if not expected_hash or actual_hash != expected_hash:
    raise SystemExit("cached component checksum does not match export metadata")
test_ids = pd.read_csv("data/test.csv")["id"].to_numpy(dtype=np.int64)
with np.load(component_path) as saved:
    required = {"id", "catboost", "context", "hierarchical", "candidate"}
    if set(saved.files) != required:
        raise SystemExit("cached component fields do not match the contract")
    component_ids = np.asarray(saved["id"], dtype=np.int64)
    if not np.array_equal(component_ids, test_ids):
        raise SystemExit("cached component IDs do not match data/test.csv order")
    values = light70_blend(
        np.asarray(saved["catboost"], dtype=np.float64),
        np.asarray(saved["context"], dtype=np.float64),
        np.asarray(saved["hierarchical"], dtype=np.float64),
    )
result = route._write_submission(
    Path("restore-output/submission_slowcat_route_context_hierarchical_30_35_35_fallback.csv"),
    test_ids,
    values,
)
expected = "9b8bee3c3290ac93472b845f586363b53f805767451845411a99fde4b165acaa"
if result["sha256"] != expected:
    raise SystemExit(f"unexpected fallback CSV checksum: {result['sha256']}")
print(f"fallback recovered {result['rows']} rows; sha256={result['sha256']}")
PY
```

The fallback must produce the same expected SHA-256 as the native reload. The
cached `candidate` field is the historical 50/25/25 blend and must not be used
as the final submission.

## Verified restore result and limits

On **2026-11-01 UTC**, a separate local clone at source commit
`4c73e90ba40a0dab9221956c3dd0cc64938ee083` restored only the 13 listed files
(60,869,255 bytes, about 61 MB). The clone used a fresh Python 3.10.22
environment installed from `requirements-dev.txt` with `uv pip install`.
Linux x86-64, glibc 2.39 and the original VPS CPU/OS were unchanged.
Core numerical dependencies were NumPy 1.26.4, pandas 2.2.3, SciPy 1.14.1,
scikit-learn 1.5.2, LightGBM 4.6.0 and CatBoost 1.2.8.

The guide's backup, manifest verification, working-copy path rebase, native
reload, CSV writing, and cached-fallback command blocks were executed. Both
native inference and cached recovery reproduced all **299,844** rows and SHA-256
`9b8bee3c3290ac93472b845f586363b53f805767451845411a99fde4b165acaa`.
All three native component prediction arrays matched the saved arrays exactly.
The documented verifier also rejected a corrupted payload and an extra file
not covered by the manifest. The original archived files stayed unchanged.

This verifies relocation and a fresh dependency installation on the same
host, not a physically different machine or OS. On a different machine, run
the commands and require the final checksum to match; report any mismatch,
rather than silently treating approximately equal predictions as identical.
The cached fallback preserves the exact original outcome if native inference
changes across platforms.

A slim snapshot does **not** enable rebuilding every historical experiment or
retraining the final models from raw inputs alone. That requires separate
pipeline work. Historical timestamps, timings and native model identifiers
cannot be regenerated byte-for-byte by these instructions. If “everything”
means the entire original artifact tree, archive that full tree instead; do
not delete historical files while expecting the slim snapshot to recover them.
No external backup/upload or local project deletion was performed by this test.

Run the project quality gates from the repository root after the restore
procedure. They are documented here for the parent workflow and are not part
of the portable model archive:

```bash
.venv/bin/ruff format --check --exclude 'artifacts/**' .
.venv/bin/ruff check . --output-format=concise
.venv/bin/mypy --check-untyped-defs *.py
.venv/bin/python -m unittest discover -q
git diff --check
```
