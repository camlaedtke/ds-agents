"""The split manifest fits inside `read_artifact`'s cap for every manifest dataset, with room to
spare: `feature_eng` refuses on a truncated read, but the run continues and writes a publishable
row with no model in it, so a dataset that cannot be run would not be loud, it would look like a
measurement.

No CSV is read. Only the SIZE of the manifest matters, and that is fixed by the agent row count:
the `assignment` field is exactly one character per agent row, and everything else in the manifest
is a small, near-constant header. See `ds_agents/split_manifest.py` for the encoding -- this file
does not reimplement it, it builds real manifests through `manifest_from` (tests/conftest.py) and
measures what comes out.
"""

import json

import numpy as np
import pytest

from ds_agents.benchmark import load_manifest
from ds_agents.nodes.profiler import HOLDOUT_FRACTION, N_FOLDS
from ds_agents.runnable import WITHHELD_FRACTION
from ds_agents.tools.protocol import DEFAULT_READ_BYTES
from tests.conftest import manifest_from

pytestmark = pytest.mark.fast

# How close to the cap a runnable dataset is allowed to sit: tight enough to catch a regression
# that eats most of the slack, not just one that eats literally all of it.
SAFE_FRACTION = 0.25


def project_split_manifest_bytes(n_rows: int, *, seed: int = 20260822) -> int:
    """Bytes the profiler's split manifest would occupy for a dataset of `n_rows` rows.

    Builds a REAL manifest via `manifest_from` (tests/conftest.py) and returns its serialized
    size -- not a hand-mirrored structure -- so this measures the one implementation's output rather
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
    manifest = manifest_from(
        n_rows=agent_rows,
        holdout=holdout,
        fold_valid=fold_valid,
        strategy="stratified",
        seed=seed,
        target="placeholder",
        holdout_fraction=HOLDOUT_FRACTION,
    )
    return len(json.dumps(manifest))


def test_every_manifest_dataset_fits_under_the_safe_fraction():
    """The largest projected manifest, over every dataset in the manifest, must leave
    `SAFE_FRACTION` of headroom under the read cap -- not just squeak under it."""
    entries = load_manifest().datasets
    largest = max(project_split_manifest_bytes(entry.n_rows) for entry in entries)
    assert largest < DEFAULT_READ_BYTES * SAFE_FRACTION, (
        f"the largest projected manifest is {largest:,} bytes, "
        f"{largest / DEFAULT_READ_BYTES:.1%} of the {DEFAULT_READ_BYTES:,}-byte cap -- above the "
        f"{SAFE_FRACTION:.0%} pinned headroom."
    )


def test_the_encoding_costs_one_byte_per_agent_row():
    """Guards the encoding itself: at one byte per row the cap is about a million agent rows away,
    far past any manifest dataset. Two bytes per row would halve that without failing the test
    above."""
    small, large = 1_000_000, 2_000_000
    agent_delta = (large - small) - int(round((large - small) * WITHHELD_FRACTION))
    per_row = (
        project_split_manifest_bytes(large) - project_split_manifest_bytes(small)
    ) / agent_delta
    assert per_row == pytest.approx(1.0, abs=0.01)
