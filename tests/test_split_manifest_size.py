"""The split manifest fits inside `read_artifact`'s cap for every dataset in the manifest, with
room to spare.

It used to not. The profiler wrote train, holdout and five folds of (train, valid) as explicit
row-index lists -- roughly six times the agent row count in integers -- and `read_artifact` caps
every read at `DEFAULT_READ_BYTES`. Above ~36k rows those collided: four of the thirteen benchmark
datasets (`adult`, `bank_marketing`, `higgs`, `numerai28_6`) could not complete a run. `feature_eng`
refused on the truncated read, but nothing branched on `recoverable`, so the run spent a full run's
tokens and still wrote a publishable row with no model in it -- a dataset that cannot be run was not
loud, it looked like a measurement.

That is fixed: the profiler now writes a one-character-per-row fold-assignment string (`"h"` for
holdout, `"0".."4"` for the fold a row validates in) instead of index lists. See
`ds_agents/split_manifest.py` for the encoding -- this file does not reimplement it. It builds real
manifests through `split_manifest.manifest_from` and measures what comes out, which is what stops
`project_split_manifest_bytes` from being a second, hand-typed answer to the same question.

Two things changed alongside the fix, both worth knowing here:

1. `graph.py`'s edges now halt straight to `reporter` when the state carries an error with
   `recoverable=False` (`nodes/router.py:halt_or`), and `results_row()` gained a `halted_at`
   column. So if a manifest ever got too big again, it would land as a loud `halted_at` row, not a
   silent one with no model in it. These tests exist to catch the regression before that fallback
   is ever needed, not because it isn't there.
2. Headroom is enormous now. The new encoding is `n_rows` bytes plus a small header, so the cap
   isn't reached until somewhere around a million agent rows --
   `test_the_new_cliff_sits_around_a_million_agent_rows` pins the actual number.

No CSV is read. Only the SIZE of the manifest matters, and that is fixed by the agent row count:
the `assignment` field is exactly one character per agent row, and everything else in the manifest
is a small, near-constant header.
"""

import json

import numpy as np
import pytest

from ds_agents import split_manifest
from ds_agents.benchmark import load_manifest
from ds_agents.harness import SUBSETS
from ds_agents.nodes.profiler import HOLDOUT_FRACTION, N_FOLDS
from ds_agents.runnable import WITHHELD_FRACTION
from ds_agents.tools.protocol import DEFAULT_READ_BYTES

pytestmark = pytest.mark.fast

# The set of datasets that exceed the cap, as of 2026-09-01. It used to be
# {"adult", "bank_marketing", "higgs", "numerai28_6"}; it is empty now that the split manifest is a
# fold-assignment string instead of index lists. Kept as a named, asserted-empty set rather than a
# bare `set()` literal in the test below, so the next regression fails a comparison against a name
# that says what moved, not just an inequality.
KNOWN_UNRUNNABLE: set[str] = set()

# How close to the cap a runnable dataset is allowed to sit. The old value, 0.95, was tuned to
# `nomao` at 0.87 -- there was no room to demand more headroom without excluding a dataset that
# actually ran. Under the new encoding the worst case (`higgs`) sits at 0.075, so 0.25 is still 3x
# headroom over the worst real dataset while being tight enough to catch a regression that eats
# most of the new slack, rather than only one that eats literally all of it.
SAFE_FRACTION = 0.25


def project_split_manifest_bytes(n_rows: int, *, seed: int = 20260822) -> int:
    """Bytes the profiler's split manifest would occupy for a dataset of `n_rows` rows.

    Builds a REAL manifest via `split_manifest.manifest_from` and returns its serialized size --
    not a hand-mirrored structure -- so this measures the one implementation's output rather than
    offering a second answer to what the manifest looks like. It withholds `WITHHELD_FRACTION`
    before the graph starts, carves `HOLDOUT_FRACTION` off what remains, and splits the rest into
    `N_FOLDS` folds, the same shape `nodes/profiler.py` produces.
    """
    agent_rows = n_rows - int(round(n_rows * WITHHELD_FRACTION))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(agent_rows)
    n_holdout = int(round(agent_rows * HOLDOUT_FRACTION))
    holdout = shuffled[:n_holdout].tolist()
    train = shuffled[n_holdout:]
    parts = np.array_split(rng.permutation(train), N_FOLDS)
    fold_valid = [part.tolist() for part in parts]
    manifest = split_manifest.manifest_from(
        n_rows=agent_rows,
        holdout=holdout,
        fold_valid=fold_valid,
        strategy="stratified",
        seed=seed,
        target="placeholder",
        holdout_fraction=HOLDOUT_FRACTION,
    )
    return len(json.dumps(manifest))


def _agent_manifest_bytes(agent_rows: int, *, seed: int = 20260822) -> int:
    """Bytes a manifest of exactly `agent_rows` agent rows would occupy.

    Used only by the cliff test below. Unlike `project_split_manifest_bytes` this works directly in
    agent-row space (no `WITHHELD_FRACTION` conversion) and partitions rows by plain slicing rather
    than a shuffle -- the cliff test only needs the row count where the byte total crosses the cap,
    and `split_manifest.py`'s docstring notes that which particular row lands in which fold moves
    the byte count by a fraction of a percent, nowhere near enough to move a million-row answer.
    """
    n_holdout = int(round(agent_rows * HOLDOUT_FRACTION))
    holdout = list(range(n_holdout))
    train = list(range(n_holdout, agent_rows))
    fold_valid: list[list[int]] = [[] for _ in range(N_FOLDS)]
    for i, row in enumerate(train):
        fold_valid[i % N_FOLDS].append(row)
    manifest = split_manifest.manifest_from(
        n_rows=agent_rows,
        holdout=holdout,
        fold_valid=fold_valid,
        strategy="stratified",
        seed=seed,
        target="placeholder",
        holdout_fraction=HOLDOUT_FRACTION,
    )
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


def test_every_manifest_dataset_now_fits_under_the_read_cap():
    """Pins the finding's reversal: the set of datasets that cannot be run is empty.

    Was `test_four_manifest_datasets_cannot_be_run_until_the_split_manifest_shrinks`, which pinned
    `KNOWN_UNRUNNABLE = {"adult", "bank_marketing", "higgs", "numerai28_6"}`. That set is now empty,
    which fixing the split manifest was supposed to accomplish -- so this asserts it stayed empty
    AND that the closest any real dataset comes to the cap still leaves the pinned `SAFE_FRACTION`
    of headroom, so a regression that eats most (but not yet all) of the new slack is still loud.
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
    largest = max(project_split_manifest_bytes(entry.n_rows) for entry in entries)
    assert largest < DEFAULT_READ_BYTES * SAFE_FRACTION, (
        f"the largest projected manifest is {largest:,} bytes, "
        f"{largest / DEFAULT_READ_BYTES:.1%} of the {DEFAULT_READ_BYTES:,}-byte cap -- above the "
        f"{SAFE_FRACTION:.0%} pinned headroom."
    )


def test_the_new_cliff_sits_around_a_million_agent_rows():
    """Pins where the boundary is now, in agent rows, so the next person inherits a number rather
    than a hole.

    A real `manifest_from` call at ~1M rows takes well under a second, so this uses two real calls
    rather than reimplementing the byte math by hand: one at a reference size to measure the fixed
    per-manifest overhead (the `assignment` field is exactly `agent_rows` characters; everything
    else -- version, seed, counts, the sha256 -- is that overhead, and it only grows by a handful of
    bytes as `agent_rows` gains digits, which is why the reference below is chosen at the same
    digit length as the predicted answer), and one at the arithmetic prediction to confirm it lands
    within a byte of the cap. No binary search, and no repeatedly building near-million-row strings.
    """
    reference_agent_rows = 1_000_000
    overhead = _agent_manifest_bytes(reference_agent_rows) - reference_agent_rows
    cliff = DEFAULT_READ_BYTES - overhead

    assert 900_000 < cliff < 1_100_000, f"the cliff moved to {cliff:,} agent rows"
    assert _agent_manifest_bytes(cliff) >= DEFAULT_READ_BYTES
    assert _agent_manifest_bytes(cliff - 1) < DEFAULT_READ_BYTES
