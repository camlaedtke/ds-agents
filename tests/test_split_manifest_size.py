"""The split manifest is a JSON list of every row index, and `read_artifact` is capped.

Those two facts collide on the larger half of the benchmark manifest, and nothing caught it until
someone tried to price a run on `adult`. The profiler writes `train`, `holdout` and five folds of
(train, valid) as explicit index lists (`nodes/profiler.py`'s `SPLIT_SNIPPET`), so the artifact
holds roughly six times the agent row count in integers. `read_artifact` caps every read at
`DEFAULT_READ_BYTES`, and on a truncated read `feature_eng` and `modeler` both refuse with
`recoverable=False` -- but nothing in the graph branches on `recoverable`, so the run continues,
spends a full run's tokens, and writes a publishable row with no model and no score in it.

A dataset that cannot be run is therefore not loud. It is a row that looks like a measurement. That
is what these tests are here to stop, in milliseconds and for free, rather than mid-invocation.

No CSV is read. Only the SIZE of the manifest matters, and that is fixed by the agent row count:
every index appears once across train+holdout and each train index appears five more times across
the folds, so which particular index lands in which fold moves the byte count by a fraction of a
percent. Checked against the real thing on 2026-09-01: the projection below is within 0.15% of a
manifest built by actually running the profiler's split on the cached CSVs, on all nine datasets it
was compared against.
"""

import json

import numpy as np
import pytest

from ds_agents.benchmark import load_manifest
from ds_agents.harness import SUBSETS
from ds_agents.nodes.profiler import HOLDOUT_FRACTION, N_FOLDS
from ds_agents.runnable import WITHHELD_FRACTION
from ds_agents.tools.protocol import DEFAULT_READ_BYTES

pytestmark = pytest.mark.fast

# Datasets known to exceed the cap as of 2026-09-01. Pinned so that fixing the representation --
# which is what `--subset full` now waits on -- fails this test and forces the several places that
# describe the blocker to be corrected together.
KNOWN_UNRUNNABLE = {"adult", "bank_marketing", "higgs", "numerai28_6"}

# How close to the cap a runnable dataset is allowed to sit. `nomao` is at 0.87, so there is no
# room to demand more headroom than this without excluding it -- which is itself worth knowing.
SAFE_FRACTION = 0.95


def project_split_manifest_bytes(n_rows: int, *, seed: int = 20260822) -> int:
    """Bytes the profiler's split manifest would occupy for a dataset of `n_rows` rows.

    Mirrors `SPLIT_SNIPPET` structurally rather than by import: the snippet is a string executed in
    a sandbox, so there is nothing to call. It withholds `WITHHELD_FRACTION` before the graph
    starts, carves `HOLDOUT_FRACTION` off what remains, and writes `N_FOLDS` folds over the train
    rows.
    """
    agent_rows = n_rows - int(round(n_rows * WITHHELD_FRACTION))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(agent_rows)
    n_holdout = int(round(agent_rows * HOLDOUT_FRACTION))
    holdout = np.sort(shuffled[:n_holdout])
    train = np.sort(shuffled[n_holdout:])
    parts = np.array_split(rng.permutation(train), N_FOLDS)
    folds = [
        {
            "train": np.sort(np.concatenate([p for j, p in enumerate(parts) if j != i])).tolist(),
            "valid": np.sort(parts[i]).tolist(),
        }
        for i in range(N_FOLDS)
    ]
    manifest = {
        "strategy": "stratified",
        "seed": seed,
        "target": "placeholder",
        "n_rows": int(agent_rows),
        "train": train.tolist(),
        "holdout": holdout.tolist(),
        "folds": folds,
    }
    return len(json.dumps(manifest))


def _subset_datasets() -> set[str]:
    return {cell.dataset for cells in SUBSETS.values() for cell in cells}


def test_no_subset_names_a_dataset_whose_split_manifest_would_be_truncated():
    """The one that would have caught `adult` before it was funded.

    Fixtures are skipped: they are hundreds of rows and are not in the benchmark manifest.
    """
    entries = {entry.dataset_id: entry for entry in load_manifest().datasets}
    for dataset in sorted(_subset_datasets()):
        entry = entries.get(dataset)
        if entry is None:
            continue  # a fixture; sized in the hundreds of rows
        projected = project_split_manifest_bytes(entry.n_rows)
        assert projected < DEFAULT_READ_BYTES * SAFE_FRACTION, (
            f"{dataset} would write a {projected:,}-byte split manifest against a "
            f"{DEFAULT_READ_BYTES:,}-byte read cap. feature_eng refuses on a truncated read, but "
            "the run continues and writes a publishable row with no model in it -- so this would "
            "be paid for and look like a measurement."
        )


def test_four_manifest_datasets_cannot_be_run_until_the_split_manifest_shrinks():
    """Pins the finding itself, and the size of it.

    `--subset full` is 13 cells and four of them cannot produce a row today. That is a SECOND
    blocker beside the measured-cost one, and it is code rather than money. When someone changes
    how the split is represented, this test fails and points at every message that says otherwise.
    """
    entries = load_manifest().datasets
    over = {
        entry.dataset_id
        for entry in entries
        if project_split_manifest_bytes(entry.n_rows) >= DEFAULT_READ_BYTES
    }
    assert over == KNOWN_UNRUNNABLE, (
        f"the set of unrunnable datasets moved: {sorted(over)} against a pinned "
        f"{sorted(KNOWN_UNRUNNABLE)}"
    )


def test_the_cliff_sits_between_the_largest_runnable_and_the_smallest_broken():
    """Where the boundary is, in rows, so the next person does not have to rediscover it."""
    entries = {entry.dataset_id: entry for entry in load_manifest().datasets}
    largest_ok = entries["nomao"].n_rows
    smallest_broken = entries["bank_marketing"].n_rows
    assert largest_ok < smallest_broken
    assert project_split_manifest_bytes(largest_ok) < DEFAULT_READ_BYTES
    assert project_split_manifest_bytes(smallest_broken) >= DEFAULT_READ_BYTES
