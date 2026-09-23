"""Builds `evals/datasets/manifest.yaml`. The only thing in the repo that writes that file.

`.claude/settings.json` denies Edit and Write under `evals/datasets/`, so the manifest cannot be
typed -- it can only be produced by running this. That is deliberate. See `benchmark.py`'s module
docstring for why a hand-written manifest is the specific failure this architecture prevents.

Everything here does network I/O and reads CSVs, which is why it is separate from `benchmark.py`:
importing the registry to read the manifest must not drag in sklearn, urllib, or a download.

Sources, all clean HTTPS and all re-checkable:
  - the AMLB dataset lists, at a pinned immutable commit
  - OpenML's task API, for the task -> data id mapping and the task's target feature
  - OpenML's data API, for name, version, format, url and the upstream md5
  - OpenML's evaluation API, for a published score citable to a stable run id
  - the frames themselves, via `sklearn.datasets.fetch_openml`, for every measured number
"""

import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
import yaml

from ds_agents.benchmark import (
    AMLB_REF,
    BASELINE_DEFINITION,
    CITATION,
    DATASET_CACHE,
    KNOWN_LEAKAGE,
    MANIFEST_PATH,
    OPENML_CACHE,
    SCHEMA_VERSION,
    SELECTION_RULE,
    DatasetEntry,
    DatasetManifest,
    ExcludedDataset,
    GeneratedWith,
    PublishedReference,
)

# Imported rather than restated, so `min_usable_features` cannot drift away from the filters it is
# meant to model. The ban is one-directional: `nodes/` must not import a registry, but a build
# script reading a node's constants is fine, and `tests/test_fixture_registry.py` already does it.
from ds_agents.nodes.feature_eng import ID_DISTINCTNESS_THRESHOLD, MAX_ONE_HOT_LEVELS
from ds_agents.provenance import file_sha256

TIMEOUT_S = 60
AMLB_RAW = "https://raw.githubusercontent.com/openml/automlbenchmark/{ref}/{path}"
OPENML_TASK = "https://www.openml.org/api/v1/json/task/{task_id}"
OPENML_DATA = "https://www.openml.org/api/v1/json/data/{data_id}"
# How many uploaded evaluations to consider when picking the best published score. 500 is well
# past the count on any task in this set; it exists so the request is bounded, not to sample.
EVAL_LIST_LIMIT = 500
OPENML_EVAL = (
    "https://www.openml.org/api/v1/json/evaluation/list/task/{task_id}"
    "/function/{function}/limit/{limit}"
)
OPENML_RUN_URL = "https://www.openml.org/r/{run_id}"
# OpenML's name for roc_auc. The manifest records our `Metric` literal, not this one.
OPENML_AUC = "area_under_roc_curve"


def _get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return json.load(response)


def _get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return response.read().decode("utf-8")


def _dataset_id(openml_name: str) -> str:
    """Our name for a dataset: snake_case, and stable.

    Deliberately ours rather than OpenML's, because `--dataset` has to name fixtures and benchmark
    datasets in one namespace and `blood-transfusion-service-center` is not a thing anyone types.
    """
    out = []
    for char in openml_name:
        out.append(char.lower() if (char.isalnum()) else "_")
    collapsed = "_".join(part for part in "".join(out).split("_") if part)
    return collapsed


@dataclass(frozen=True)
class Candidate:
    """One AMLB list entry, before anything about it has been measured."""

    openml_name: str
    openml_task_id: int
    amlb_list: str


def amlb_candidates(ref: str = AMLB_REF) -> list[Candidate]:
    """Every entry in AMLB's small and medium lists at the pinned commit."""
    candidates: list[Candidate] = []
    for list_name, path in CITATION.amlb_lists.items():
        entries = yaml.safe_load(_get_text(AMLB_RAW.format(ref=ref, path=path)))
        for entry in entries:
            candidates.append(
                Candidate(
                    openml_name=str(entry["name"]),
                    openml_task_id=int(entry["openml_task_id"]),
                    amlb_list=list_name,
                )
            )
    return sorted(candidates, key=lambda c: (c.openml_name.lower(), c.openml_task_id))


def published_reference(task_id: int) -> PublishedReference | None:
    """The best published AUC on this OpenML task, with the run id that produced it.

    Informational, and `protocol` says so on the value itself. `max` rather than a mean because
    the useful anchor for "is this dataset learnable, and roughly how well" is what the best
    uploaded flow achieved; a mean over crowd-sourced runs mostly measures who uploaded.

    `None` rather than a guess when the task has no evaluations. A missing anchor is a fact; an
    invented one is the failure this whole module is built to prevent.
    """
    try:
        payload = _get_json(
            OPENML_EVAL.format(task_id=task_id, function=OPENML_AUC, limit=EVAL_LIST_LIMIT)
        )
    except Exception as error:  # noqa: BLE001 -- a fetch failure is data, not a crash
        # Logged rather than swallowed. A silent `None` here is indistinguishable from "OpenML has
        # no evaluations for this task", so a transient timeout would quietly drop a real citable
        # score and the manifest would understate its own provenance with no trace that it had.
        print(
            f"  ! task {task_id}: no published reference ({type(error).__name__}: {error})",
            file=sys.stderr,
        )
        return None
    evaluations = payload.get("evaluations", {}).get("evaluation", [])
    scored = [e for e in evaluations if e.get("value") not in (None, "")]
    if not scored:
        return None
    best = max(scored, key=lambda e: float(e["value"]))
    return PublishedReference(
        source="OpenML evaluation API",
        openml_run_id=int(best["run_id"]),
        openml_run_url=OPENML_RUN_URL.format(run_id=int(best["run_id"])),
        flow_name=str(best.get("flow_name", "")),
        metric="roc_auc",
        value=round(float(best["value"]), 6),
        protocol=(
            "OpenML task estimation procedure (10-fold cross-validation), run by a third-party "
            "flow. NOT this repo's withheld holdout -- informational only, never baseline_score."
        ),
    )


def _n_usable_features(frame: pd.DataFrame, target: str) -> int:
    """How many columns would still be standing when the modeler sees the matrix.

    Reproduces `feature_eng`'s two mechanical filters using its own constants: an identifier-like
    column (non-float dtype, at least ID_DISTINCTNESS_THRESHOLD distinct) is dropped, and a
    categorical above MAX_ONE_HOT_LEVELS levels is skipped by the encoder.
    """
    n_rows = len(frame)
    usable = 0
    for name in frame.columns:
        if name == target:
            continue
        column = frame[name]
        is_float = pd.api.types.is_float_dtype(column)
        n_unique = int(column.nunique(dropna=True))
        if not is_float and n_rows > 0 and n_unique >= ID_DISTINCTNESS_THRESHOLD * n_rows:
            continue
        if not pd.api.types.is_numeric_dtype(column) and n_unique > MAX_ONE_HOT_LEVELS:
            continue
        usable += 1
    return usable


def _write_csv(frame: pd.DataFrame, path: Path) -> str:
    """Write the canonical CSV deterministically and return its sha256.

    Explicit about every option that pandas would otherwise decide by version: a byte-identity
    claim is only worth making if it is reproducible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="", lineterminator="\n")
    return file_sha256(path)


@dataclass
class Measured:
    """A candidate after fetching: everything the selection rule needs to judge it."""

    candidate: Candidate
    entry: DatasetEntry | None
    measured: dict[str, float | int | str]
    failed: str | None


def measure(candidate: Candidate) -> Measured:
    """Fetch one candidate and measure it. Never applies the rule -- that is `select`."""
    from sklearn.datasets import fetch_openml

    task = _get_json(OPENML_TASK.format(task_id=candidate.openml_task_id))["task"]
    source = next(i for i in task["input"] if i["name"] == "source_data")["data_set"]
    data_id = int(source["data_set_id"])
    # The TASK's target, not the dataset's default: the task is what AMLB pins, and the two can
    # disagree (Australian's task names A15 while the data description names its own default).
    target = str(source["target_feature"])

    description = _get_json(OPENML_DATA.format(data_id=data_id))["data_set_description"]

    OPENML_CACHE.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=str(OPENML_CACHE), parser="auto")
    frame = bunch.frame
    if target not in frame.columns:
        return Measured(candidate, None, {"target": target}, "target_not_in_frame")

    # n_classes is a property of the data, not of any encoding, so it is safe to screen on the
    # in-memory frame -- and screening here avoids writing a 70000 x 784 CSV for a 10-class
    # dataset we are about to reject anyway.
    n_classes = int(frame[target].nunique(dropna=True))
    n_rows = int(len(frame))
    n_features = int(frame.shape[1] - 1)
    measured: dict[str, float | int | str] = {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_classes": n_classes,
    }
    if n_classes != 2:
        return Measured(candidate, None, measured, "task_type")

    dataset_id = _dataset_id(candidate.openml_name)
    csv_path = DATASET_CACHE / dataset_id / f"{dataset_id}.csv"
    csv_sha256 = _write_csv(frame, csv_path)

    # Everything below is measured on the CSV AS RE-READ, not on the frame that was fetched. The
    # CSV is what gets mounted at $DS_DATASET, so it is what the pipeline actually sees, and the
    # round trip is not lossless: kc1's target arrives from OpenML as the category levels 'true'
    # and 'false' and comes back out of pandas as the bools True and False. A manifest that
    # recorded 'true' would name a positive class no run could ever match, and `positive_rate`
    # would describe a file nobody reads. Dtypes shift the same way, which also moves
    # `n_usable_features` -- the number that decides whether the matrix reaches the modeler.
    frame = pd.read_csv(csv_path, low_memory=False)

    labels = frame[target]
    counts = labels.astype(str).value_counts(dropna=True)
    # The minority level, named explicitly. `positive_rate` is meaningless without saying which
    # class it counts, and "the second one alphabetically" is not a definition anyone can rely on.
    positive_class = str(counts.idxmin())
    positive_rate = round(float((labels.astype(str) == positive_class).mean()), 6)
    usable = _n_usable_features(frame, target)
    measured["n_usable_features"] = usable
    measured["positive_rate"] = positive_rate

    # A hand-authored `known_leakage` entry must name a column that actually exists in the frame
    # the pipeline will see, and this check exists because it caught a real one: `bank_marketing`
    # was first recorded against `duration`, the name UCI uses, but OpenML data 1461 ships
    # anonymised headers (V1..V16) and no such column exists. Loud failure at build time, because
    # a leak entry pointing at a phantom column is worse than no entry -- it reads as documented
    # ground truth and can never fire.
    known = KNOWN_LEAKAGE.get(dataset_id, [])
    missing = [leak.column for leak in known if leak.column not in frame.columns]
    if missing:
        raise ValueError(
            f"{dataset_id}: KNOWN_LEAKAGE names column(s) {missing} that are not in the fetched "
            f"frame. Columns are: {list(frame.columns)}"
        )

    entry = DatasetEntry(
        dataset_id=dataset_id,
        openml_data_id=data_id,
        openml_task_id=candidate.openml_task_id,
        openml_name=str(description["name"]),
        openml_version=int(description["version"]),
        openml_format=str(description["format"]),
        openml_url=str(description["url"]),
        md5_checksum=str(description["md5_checksum"]),
        amlb_list=candidate.amlb_list,
        target=target,
        task_type="binary",
        metric=SELECTION_RULE.metric,
        positive_class=positive_class,
        n_rows=n_rows,
        n_features=n_features,
        n_usable_features=usable,
        positive_rate=positive_rate,
        csv_sha256=csv_sha256,
        leakage_labelled=False,
        known_leakage=known,
        published_reference=published_reference(candidate.openml_task_id),
    )
    return Measured(candidate, entry, measured, None)


def select(results: list[Measured]) -> tuple[list[DatasetEntry], list[ExcludedDataset]]:
    """Apply `SELECTION_RULE` to what was MEASURED. Never to what was expected."""
    kept: list[Measured] = []
    excluded: list[ExcludedDataset] = []

    def reject(item: Measured, failed: str) -> None:
        excluded.append(
            ExcludedDataset(
                openml_name=item.candidate.openml_name,
                openml_data_id=item.entry.openml_data_id if item.entry else -1,
                openml_task_id=item.candidate.openml_task_id,
                amlb_list=item.candidate.amlb_list,
                failed=failed,
                measured=item.measured,
            )
        )

    for item in results:
        if item.entry is None:
            reject(item, item.failed or "unmeasurable")
            continue
        entry = item.entry
        if entry.n_rows > SELECTION_RULE.max_rows:
            reject(item, "max_rows")
        elif entry.n_features > SELECTION_RULE.max_features:
            reject(item, "max_features")
        elif entry.n_rows * entry.n_features > SELECTION_RULE.max_cells:
            reject(item, "max_cells")
        elif entry.n_usable_features < SELECTION_RULE.min_usable_features:
            reject(item, "min_usable_features")
        else:
            kept.append(item)

    _, upper = SELECTION_RULE.target_count
    if len(kept) > upper:
        kept.sort(key=lambda i: (i.entry.n_rows * i.entry.n_features, i.entry.dataset_id))  # type: ignore[union-attr]
        for item in kept[upper:]:
            reject(item, "target_count")
        kept = kept[:upper]

    entries = sorted((i.entry for i in kept if i.entry), key=lambda e: e.dataset_id)
    excluded.sort(key=lambda e: (e.failed, e.openml_name.lower()))
    return entries, excluded


def build() -> DatasetManifest:
    """Fetch every candidate, measure it, apply the rule, and return the manifest."""
    candidates = amlb_candidates()
    results = []
    for candidate in candidates:
        try:
            results.append(measure(candidate))
        except Exception as error:  # noqa: BLE001 -- a fetch failure is data, not a crash
            print(f"  ! {candidate.openml_name}: {type(error).__name__}: {error}", file=sys.stderr)
            results.append(
                Measured(candidate, None, {"error": f"{type(error).__name__}"}, "fetch_failed")
            )
    entries, excluded = select(results)

    source = CITATION.model_copy(
        update={
            "generated_with": GeneratedWith(
                python=f"{sys.version_info.major}.{sys.version_info.minor}",
                pandas=pd.__version__,
                scikit_learn=sklearn.__version__,
            )
        }
    )
    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        source=source,
        baselines=BASELINE_DEFINITION,
        selection=SELECTION_RULE,
        datasets=entries,
        excluded=excluded,
    )


HEADER = """\
# GENERATED FILE -- do not edit by hand.
#
# Written by `uv run ds-agents datasets refresh`; checked by `uv run ds-agents datasets verify
# --online`, which re-fetches, regenerates, and diffs. .claude/settings.json denies Edit and Write
# under evals/datasets/, and that deny is what makes the following claim true: every number below
# is either a field of an API response or a measurement taken on the fetched frame. Nothing here
# was typed by a person. The prose lives in src/ds_agents/benchmark.py, where it is reviewable.
#
# `published_reference` is INFORMATIONAL. It was produced on OpenML's own 10-fold cross-validation
# by a third-party flow, so it is not comparable to this pipeline's verified_holdout_score and must
# never be assigned to PipelineState.baseline_score. See `baselines` below for what baseline_score
# is to be computed from instead.
"""


def _quoted_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Emit every string quoted.

    YAML 1.1 resolves an unquoted `2026-08-31` to a date and an unquoted `no`, `yes`, `on` or `off`
    to a bool -- and a column named `no` is not hypothetical in OpenML data. Quoting everything
    means a round trip through `safe_load` returns what was written.
    """
    style = "|" if "\n" in data else "'"
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, _quoted_str)


def render(manifest: DatasetManifest) -> str:
    """The manifest as the exact text that goes on disk. Pure, so `verify` can diff against it."""
    payload = manifest.model_dump(mode="json", exclude_none=True)
    body = yaml.dump(
        payload, Dumper=_Dumper, default_flow_style=False, sort_keys=False, width=98, indent=2
    )
    return HEADER + "\n" + body


def refresh(path: Path = MANIFEST_PATH) -> DatasetManifest:
    """Build the manifest and write it. The only write to `evals/datasets/` in the repo."""
    manifest = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(manifest))
    return manifest
